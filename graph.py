import json
import os
import re
from typing import Dict, List, Any
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

from validator import MemoryValidator
from empathy import EmpathyFilter
from vector_memory import VectorMemory

# ============ 加载环境变量 ============
load_dotenv()

# ============ 配置区 ============
DB_PATH = "agent_memory.db"
validator = MemoryValidator(db_path=DB_PATH)
vector_memory = VectorMemory()
empathy = EmpathyFilter(vector_memory=vector_memory)

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
    recent_raw_dialog: List[str]   # 新增：原始对话兜底缓存

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
    except:
        fact = {"姓名":"","爱好":"","instant_emotion":"neutral","query_history_intent":False}
    print(f"[硬事实提取] {fact}")
    return {"fact_extract": fact}


def vector_memory_node(state: AgentState):
    txt = state["user_text"]
    # 修复：top_k改为4，扩大召回数量
    recall_list = vector_memory.search_memory(user_id="user_001", query=txt, top_k=4)
    recall_list = sorted(recall_list)
    return {"recall_memories": recall_list}


def update_emotion_node(state: AgentState):
    user_input = state["user_text"]
    instant_emotion = state["fact_extract"].get("instant_emotion", "neutral")
    # 兜底原始对话，不完全依赖向量召回
    recent_raw = state.get("recent_raw_dialog", [])
    emotion_result = empathy.update_emotion(
        user_id="user_001",
        user_input=user_input,
        instant_emotion=instant_emotion,
        retrieved_memories=state["recall_memories"],
        raw_recent_dialog=recent_raw
    )
    print(f"[瞬时情绪] {emotion_result['instant']}")
    print(f"[长期情绪画像] joy:{emotion_result['baseline']['joy']:.2f} sadness:{emotion_result['baseline']['sadness']:.2f} anger:{emotion_result['baseline']['anger']:.2f} fear:{emotion_result['baseline']['fear']:.2f}")
    print(f"[情绪偏移检测] {emotion_result['delta_desc']}")
    return {"emotion_info": emotion_result}


def generate_reply_node(state: AgentState):
    user_text = state["user_text"]
    facts = state["fact_extract"]
    recalls = state["recall_memories"]
    emo = state["emotion_info"]

    fact_str = f"用户姓名：{facts.get('姓名','无')}，爱好：{facts.get('爱好','无')}"
    mem_str = "\n".join(recalls) if recalls else "无相关历史记忆"
    baseline = emo["baseline"]
    delta_desc = emo["delta_desc"]

    sys_prompt = f"""
你是共情对话助手。
【用户硬事实】
{fact_str}

【检索到的历史情景记忆】
{mem_str}

【用户长期情绪基线】
joy:{baseline['joy']:.2f}, sadness:{baseline['sadness']:.2f}, anger:{baseline['anger']:.2f}, fear:{baseline['fear']:.2f}
【情绪偏移提示】{delta_desc}

约束：
1.只允许使用上面提供的记忆内容，禁止编造用户没有说过的经历。
2.情绪基线与偏移描述仅用来把握说话语气，不要直接输出数值。
3.如果没有记忆，如实说明，不要虚构故事。
4.说话温和有共情，简短自然。
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
workflow.add_node("update_emotion", update_emotion_node)
workflow.add_node("gen_reply", generate_reply_node)

workflow.add_edge(START, "extract_fact")
workflow.add_edge("extract_fact", "vector_retrieve")
workflow.add_edge("vector_retrieve", "update_emotion")
workflow.add_edge("update_emotion", "gen_reply")
workflow.add_edge("gen_reply", END)

app = workflow.compile()

# ============ 主交互循环 ============
def main():
    print("="*50)
    print("✅ 流式聊天已启动！输入 quit 结束对话")
    print("="*50)
    raw_dialog_buf = []  # 本地维护最近原始对话兜底
    while True:
        user_input = input("\n请输入：").strip()
        if user_input.lower() == "quit":
            print("👋结束对话")
            break
        if not user_input:
            continue
        raw_dialog_buf.append(user_input)
        # 只保留最近5轮原始对话做兜底，防止上下文过长
        if len(raw_dialog_buf) > 5:
            raw_dialog_buf.pop(0)
        init_state = {
            "user_text": user_input,
            "messages": [],
            "fact_extract": {},
            "recall_memories": [],
            "emotion_info": {},
            "recent_raw_dialog": raw_dialog_buf
        }
        try:
            result = app.invoke(init_state)
            # 节点跑完之后，后置写入向量记忆（时序隔离关键）
            vector_memory.add_memory(user_id="user_001", content=user_input)
            # print(f"[向量记忆保存] user:user_001 内容:{user_input[:40]}...")
            # 更新硬事实
            f = result.get("fact_extract",{})
            if f.get("姓名") or f.get("爱好"):
                validator.upsert_fact(user_id="user_001", name=f.get("姓名",""), hobby=f.get("爱好",""))
            recalls_out = result.get("recall_memories",[])
            print(f"[向量检索] 找到 {len(recalls_out)} 条相关情景记忆")
            print("-"*50)
        except Exception as e:
            print(f"\n🔥Graph运行异常：{e}")
            print("-"*50)

if __name__ == "__main__":
    main()
