import json
import os
import re
import time
import traceback
from typing import Dict, List, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
import hdbscan
import numpy as np


from validator import MemoryValidator
from empathy import EmpathyFilter
from vector_memory import VectorMemory


# ============ 加载环境变量 ============
load_dotenv()


# ============ 配置区 ============
DB_PATH = "agent_memory.db"
validator = MemoryValidator(db_path=DB_PATH)
vector_memory = VectorMemory()
# ✅关键：传入validator，否则HDBSCAN簇冷却逻辑不会执行
empathy = EmpathyFilter(vector_memory=vector_memory, validator=validator)


# LLM普通对话，流式开关打开
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "deepseek-chat"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0.3,
    streaming=True
)


# ============ 状态定义 ============
class AgentState(TypedDict):
    user_text: str
    messages: Annotated[list, add_messages]
    fact_extract: Dict[str, Any]
    recall_memories: List[str]
    emotion_info: Dict[str, Any]
    recent_raw_dialog: List[str]   # 原始对话兜底缓存
    cluster_id: int                # HDBSCAN输出簇id，-1噪声簇


# ============ 节点定义 ============
def extract_fact_node(state: AgentState):
    user_input = state["user_text"]
    prompt = f"""
你是信息提取器，从用户对话提取信息，输出严格json。
字段：
"姓名":字符串，没有为空字符串
"爱好":字符串，没有为空字符串
"instant_emotion":只能选 joy / sadness / anger / fear
"query_history_intent":bool，用户是否在询问自己过往记忆


用户输入：{user_input}
只输出json，不要别的内容。
"""
    resp = llm.invoke(prompt)
    raw = resp.content
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = '{"姓名":"","爱好":"","instant_emotion":"neutral","query_history_intent":false}'
    try:
        fact = json.loads(json_str)
    except Exception:
        fact = {"姓名":"","爱好":"","instant_emotion":"neutral","query_history_intent":False}
    print(f"[硬事实提取] {fact}")
    return {"fact_extract": fact}



def vector_memory_node(state: AgentState):
    txt = state["user_text"]
    recall_list = vector_memory.search_memory(user_id="user_001", query=txt, top_k=4)
    return {"recall_memories": recall_list}



def cluster_node(state: AgentState):
    """HDBSCAN局部窗口聚类，仅对内存最近N轮对话做聚类"""
    raw_dialog = state.get("recent_raw_dialog", [])
    current_text = state["user_text"]
    if len(raw_dialog) < 2:
        return {"cluster_id": -1}

    texts = raw_dialog
    emb_list = []
    for t in texts:
        emb = vector_memory.get_embedding(t)
        emb_list.append(emb)
    arr = np.array(emb_list)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1)
    clusterer.fit(arr)

    # ✅ 修复bug：不用index()，用enumerate找当前文本下标，防止重复字符串取错位置
    current_idx = None
    for idx, s in enumerate(texts):
        if s == current_text:
            current_idx = idx
            break
    if current_idx is None:
        return {"cluster_id": -1}

    cid = int(clusterer.labels_[current_idx])
    print(f"[DEBUG-CLUSTER] HDBSCAN输出cluster_id={cid}")
    return {"cluster_id": cid}



def update_emotion_node(state: AgentState):
    user_input = state["user_text"]
    instant_emotion = state["fact_extract"].get("instant_emotion", "neutral")
    recent_raw = state.get("recent_raw_dialog", [])
    emotion_result = empathy.update_emotion(
        user_id="user_001",
        user_input=user_input,
        instant_emotion=instant_emotion,
        retrieved_memories=state["recall_memories"],
        raw_recent_dialog=recent_raw,
        cluster_id=state["cluster_id"]
    )
    print(f"[瞬时情绪] {emotion_result['instant_emotion']}")
    print(f"[长期情绪画像] joy:{emotion_result['baseline']['joy']:.2f} sadness:{emotion_result['baseline']['sadness']:.2f} anger:{emotion_result['baseline']['anger']:.2f} fear:{emotion_result['baseline']['fear']:.2f}")
    print(f"[情绪偏移检测] {emotion_result['delta_desc']}")
    return {"emotion_info": emotion_result}


def _build_emotion_observation(baseline: Dict, delta_desc: str, total_score: float) -> str:
    """
    把后端情绪数值翻译为自然语言观察块，不传递浮点数给LLM
    规则：
    1. 必须 total_score >= MIN_BASELINE_TOTAL(4.0) 样本充足，才输出长期情绪倾向描述
    2. 情绪波动提示不受4.0门槛限制，只要波动大就追加
    3. 措辞保守，不使用「持续/长期」等强时序断言
    """
    obs_lines = []
    sad = baseline.get("sadness", 0.0)
    joy = baseline.get("joy", 0.0)
    anger = baseline.get("anger", 0.0)
    fear = baseline.get("fear", 0.0)

    # ✅ 只有积累足够情绪样本，才输出长期倾向
    if total_score >= EmpathyFilter.MIN_BASELINE_TOTAL:
        if sad > 0.60:
            obs_lines.append("用户难过情绪较为突出。")
        elif joy > 0.60:
            obs_lines.append("用户整体情绪比较愉悦。")
        elif anger > 0.60:
            obs_lines.append("用户当下带有烦躁生气的情绪。")
        elif fear > 0.60:
            obs_lines.append("用户当下存在不安担忧的情绪。")

    # 瞬时波动提示：不受样本门槛限制
    if "波动较大" in delta_desc:
        obs_lines.append("提示：用户当下情绪波动较大，请给予更多共情安抚。")

    if not obs_lines:
        return ""

    block = "【系统内部对用户状态的观察】\n"
    block += "\n".join(obs_lines)
    return block


def generate_reply_node(state: AgentState):
    user_text = state["user_text"]
    facts = state["fact_extract"]
    recalls = state["recall_memories"]
    emo = state["emotion_info"]
    raw_dialog = state.get("recent_raw_dialog", [])

    fact_str = f"用户姓名：{facts.get('姓名','无')}，爱好：{facts.get('爱好','无')}"
    mem_str = "\n".join(recalls) if recalls else "无相关历史记忆"
    buf_str = "\n".join([f"- {s}" for s in raw_dialog[:-1]]) if len(raw_dialog) > 1 else "暂无内存历史对话"

    baseline = emo["baseline"]
    delta_desc = emo["delta_desc"]
    total_score = sum(baseline.values())

    # ✅ 替换：不再直接把joy/sadness浮点数字塞进prompt，改为生成自然语言观察块
    emo_obs_block = _build_emotion_observation(baseline, delta_desc, total_score)

    sys_prompt = f"""
你是共情对话助手。
【用户硬事实】
{fact_str}

【检索到的历史情景记忆】
{mem_str}

【近期对话缓存（兜底，向量记忆为空时参考这里）】
{buf_str}
"""
    # 样本充足、有有效观察才拼接，空则不占token
    if emo_obs_block:
        sys_prompt += "\n" + emo_obs_block + "\n"

    sys_prompt += """
约束：
1.你必须充分阅读【近期对话缓存】，如果缓存存在历史事件，回复中要体现出你知道这些事情，不要只笼统安慰。
2.优先参考向量检索记忆；向量记忆为空时，使用【近期对话缓存】作为参考。
3.只允许使用上面提供的记忆内容，禁止编造用户没有说过的经历。
4.【系统内部对用户状态的观察】仅供把握回复语气，不要直接复述里面的文字。
5.说话温和有共情，简短自然。
"""

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_text}
    ]
    stream_resp = llm.stream(messages)
    print("\n🤖 Agent回复: ", end="", flush=True)
    full_resp = ""
    for chunk in stream_resp:
        part = chunk.content
        if part:
            print(part, end="", flush=True)
            full_resp += part
    print("\n")
    return {"messages": [{"role": "assistant", "content": full_resp}]}


# ============ 构建图 ============
workflow = StateGraph(AgentState)

workflow.add_node("extract_fact", extract_fact_node)
workflow.add_node("vector_retrieve", vector_memory_node)
workflow.add_node("cluster_node", cluster_node)
workflow.add_node("update_emotion", update_emotion_node)
workflow.add_node("gen_reply", generate_reply_node)

workflow.add_edge(START, "extract_fact")
workflow.add_edge("extract_fact", "vector_retrieve")
workflow.add_edge("vector_retrieve", "cluster_node")
workflow.add_edge("cluster_node", "update_emotion")
workflow.add_edge("update_emotion", "gen_reply")
workflow.add_edge("gen_reply", END)

app = workflow.compile()


# ============ 主交互循环 ============
def main():
    print("="*50)
    print("✅ 流式聊天已启动！输入 quit 结束对话")
    print("="*50)
    raw_dialog_buf = []  # 内存维护最近原始对话兜底
    while True:
        user_input = input("\n请输入：").strip()
        if user_input.lower() == "quit":
            print("👋结束对话")
            break
        if not user_input:
            continue

        raw_dialog_buf.append(user_input)
        # 只保留最近5轮原始对话，防止上下文过长
        if len(raw_dialog_buf) > 5:
            raw_dialog_buf.pop(0)

        vector_memory.add_memory(user_id="user_001", content=user_input)
        time.sleep(0.15)

        init_state = {
            "user_text": user_input,
            "messages": [],
            "fact_extract": {},
            "recall_memories": [],
            "emotion_info": {},
            "recent_raw_dialog": raw_dialog_buf,
            "cluster_id": -1
        }
        try:
            result = app.invoke(init_state)

            # ✅ 打开硬事实保存，适配 upsert_fact(fact_key, fact_value)
            f = result.get("fact_extract", {})
            name = f.get("姓名", "").strip()
            hobby = f.get("爱好", "").strip()
            if name:
                validator.upsert_fact(user_id="user_001", fact_key="姓名", fact_value=name)
            if hobby:
                validator.upsert_fact(user_id="user_001", fact_key="爱好", fact_value=hobby)

            recalls_out = result.get("recall_memories", [])
            print(f"[向量检索] 找到 {len(recalls_out)} 条相关情景记忆")
            print("-"*50)
        except Exception as e:
            print(f"\n🔥Graph运行异常：{e}")
            traceback.print_exc()
            print("-"*50)


if __name__ == "__main__":
    main()
