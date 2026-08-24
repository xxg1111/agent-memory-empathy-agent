import json
import os
import re
import time
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
    current_idx = texts.index(current_text)
    cid = int(clusterer.labels_[current_idx])
    print(f"[DEBUG‑CLUSTER] HDBSCAN输出cluster_id={cid}")
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
    # ✅修复key名称 instant_emotion
    print(f"[瞬时情绪] {emotion_result['instant_emotion']}")
    print(f"[长期情绪画像] joy:{emotion_result['baseline']['joy']:.2f} sadness:{emotion_result['baseline']['sadness']:.2f} anger:{emotion_result['baseline']['anger']:.2f} fear:{emotion_result['baseline']['fear']:.2f}")
    print(f"[情绪偏移检测] {emotion_result['delta_desc']}")
    return {"emotion_info": emotion_result}


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

    sys_prompt = f"""
你是共情对话助手。
【用户硬事实】
{fact_str}

【检索到的历史情景记忆】
{mem_str}

【近期对话缓存（兜底，向量记忆为空时参考这里）】
{buf_str}

【用户长期情绪基线】
joy:{baseline['joy']:.2f}, sadness:{baseline['sadness']:.2f}, anger:{baseline['anger']:.2f}, fear:{baseline['fear']:.2f}
【情绪偏移提示】{delta_desc}

约束：
1.你必须充分阅读【近期对话缓存】，如果缓存存在历史事件，回复中要体现出你知道这些事情，不要只笼统安慰。
2.优先参考向量检索记忆；向量记忆为空时，使用【近期对话缓存】作为参考。
3.只允许使用上面提供的记忆内容，禁止编造用户没有说过的经历。
4.情绪基线与偏移描述仅用来把握说话语气，不要直接输出数值。
5.说话温和有共情，简短自然。
"""
    messages = [
        {"role":"system","content":sys_prompt},
        {"role":"user","content":user_text}
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
    return {"messages": [{"role":"assistant","content":full_resp}]}

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

            # 你的validator没有upsert_fact，注释，避免报错
            # f = result.get("fact_extract",{})
            # if f.get("姓名") or f.get("爱好"):
            #     validator.upsert_fact(user_id="user_001", name=f.get("姓名",""), hobby=f.get("爱好",""))

            recalls_out = result.get("recall_memories",[])
            print(f"[向量检索] 找到 {len(recalls_out)} 条相关情景记忆")
            print("-"*50)
        except Exception as e:
            print(f"\n🔥Graph运行异常：{e}")
            print("-"*50)

if __name__ == "__main__":
    main()
