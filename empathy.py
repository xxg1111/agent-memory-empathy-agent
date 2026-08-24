# empathy.py
import sqlite3
import numpy as np
import os
import time
from langchain_openai import ChatOpenAI

class EmpathyFilter:
    def __init__(self, db_path: str = "agent_memory.db", vector_memory=None, validator=None):
        self.db_path = db_path
        self.validator = validator
        self.vector_memory = vector_memory
        self.VALID_EMOTIONS = {"joy", "sadness", "anger", "fear", "neutral"}
        self.DECAY_FACTOR = 0.94
        self.CLUSTER_COOLDOWN_SEC = 300
        self.SIM_THRESHOLD = 0.65
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        return conn

    def _init_db(self):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute('''
        CREATE TABLE IF NOT EXISTS user_emotion (
            user_id TEXT PRIMARY KEY,
            joy REAL DEFAULT 0.0,
            sadness REAL DEFAULT 0.0,
            anger REAL DEFAULT 0.0,
            fear REAL DEFAULT 0.0,
            last_update_time REAL
        )
        ''')
        conn.commit()
        conn.close()

    def _load_emotion(self, user_id: str):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute('SELECT joy,sadness,anger,fear,last_update_time FROM user_emotion WHERE user_id=?',(user_id,))
        row = cur.fetchone()
        conn.close()
        if row is None:
            return {"joy":0.0,"sadness":0.0,"anger":0.0,"fear":0.0,"last_update_time":0.0}
        return {
            "joy":row[0],
            "sadness":row[1],
            "anger":row[2],
            "fear":row[3],
            "last_update_time":row[4]
        }

    def _save_emotion(self, user_id: str, emo_dict):
        ts = time.time()
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute('''
        INSERT OR REPLACE INTO user_emotion
        (user_id,joy,sadness,anger,fear,last_update_time)
        VALUES (?,?,?,?,?,?)
        ''',(user_id,emo_dict["joy"],emo_dict["sadness"],emo_dict["anger"],emo_dict["fear"],ts))
        conn.commit()
        conn.close()

    def update_emotion(self, user_id: str, user_input: str, instant_emotion: str, retrieved_memories:list, raw_recent_dialog:list=None):
        """
        :param user_id: 用户id
        :param user_input: 当前用户输入文本
        :param instant_emotion: 瞬时情绪joy/sadness/anger/fear/neutral
        :param retrieved_memories: 向量检索返回记忆列表
        :param raw_recent_dialog: 最近原始对话兜底，防止向量召回为空
        :return: dict {"baseline":dict,"instant":str,"delta_desc":str}
        """
        now_ts = time.time()
        profile = self._load_emotion(user_id)

        # 时间衰减
        for k in ["joy","sadness","anger","fear"]:
            profile[k] *= self.DECAY_FACTOR

        if instant_emotion not in self.VALID_EMOTIONS:
            self._save_emotion(user_id, profile)
            return {
                "baseline":profile,
                "instant":"neutral",
                "delta_desc":"瞬时情绪标签非法，跳过情绪更新"
            }
        if instant_emotion == "neutral":
            self._save_emotion(user_id, profile)
            return {
                "baseline":profile,
                "instant":"neutral",
                "delta_desc":"本轮无明显情绪波动，仅执行情绪衰减"
            }

        weight = 1.0
        is_same_event = False

        # 合并两路参考：向量检索记忆 + 原始对话兜底，去重，过滤当前输入
        candidate_texts = []
        if retrieved_memories:
            candidate_texts.extend(retrieved_memories)
        if raw_recent_dialog:
            candidate_texts.extend(raw_recent_dialog)
        # 去重
        candidate_texts = list(dict.fromkeys(candidate_texts))
        # 过滤掉当前输入
        filtered = []
        for t in candidate_texts:
            clean_t = t.strip().replace("...","")
            if clean_t != user_input.strip():
                filtered.append(t)

        if len(filtered) > 0:
            ref_text = "\n".join(filtered[:3])
            prompt = f"""
判断用户新输入和下面历史片段是否描述**同一件现实事件**。
仅输出true或者false，禁止输出其它任何文字。
历史片段：
{ref_text}
用户新输入：{user_input}
true：反复复述、吐槽同一件已经发生的具体事情；
false：属于另外一件全新发生的事。
"""
            try:
                llm_judge = ChatOpenAI(
                    model=os.getenv("MODEL_NAME"),
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL"),
                    temperature=0
                )
                resp = llm_judge.invoke(prompt)
                ans = resp.content.strip().lower()
                is_same_event = (ans == "true")
            except Exception as e:
                print(f"[DEBUG‑SIM] judge error:{e}")
                is_same_event = False
        else:
            print("[DEBUG‑SIM] 无历史候选文本，判定为全新事件")
            is_same_event = False

        if is_same_event:
            weight = 0.2
            print(f"[DEBUG‑SIM] LLM判定：复述同一事件，降权 weight=0.2")
        else:
            print(f"[DEBUG‑SIM] LLM判定：全新事件，完整权重 weight=1.0")

        # HDBSCAN簇逻辑（本版本graph暂时不传cluster_id，保留接口留给后续扩展）
        cluster_id = -1
        if cluster_id != -1 and self.validator is not None:
            print(f"[DEBUG‑EMPATHY] 命中已有事件簇 cluster_id={cluster_id}")
            weight = 0.2
            last_cid_ts = self.validator.get_cluster_last_ts(cluster_id)
            if last_cid_ts is not None:
                gap = now_ts - last_cid_ts
                print(f"[DEBUG‑EMPATHY] 簇距离上次更新时间 {gap:.1f}s，冷却阈值 {self.CLUSTER_COOLDOWN_SEC}s")
                if gap > self.CLUSTER_COOLDOWN_SEC:
                    weight = 0.6
            self.validator.set_cluster_last_ts(cluster_id, now_ts)
        else:
            if not is_same_event:
                print(f"[DEBUG‑EMPATHY] 全新事件簇 cluster_id=-1，完整权重 weight=1.0")

        print(f"[DEBUG‑EMPATHY] 最终使用权重 weight={weight}")
        profile[instant_emotion] += weight
        self._save_emotion(user_id, profile)

        total = profile["joy"] + profile["sadness"] + profile["anger"] + profile["fear"]
        if total < 4.0:
            delta_desc = "暂无足够历史，重点关注本轮当下情绪"
        else:
            # 简单情绪偏移提示
            joy_val = profile["joy"]
            sad_val = profile["sadness"]
            if joy_val > 1.2 and instant_emotion == "sadness":
                delta_desc = "检测到情绪偏移：用户平时整体偏乐观，本轮出现明显负面情绪"
            elif sad_val >1.2 and instant_emotion == "joy":
                delta_desc = "检测到情绪偏移：用户近期长期处于负面状态，本轮出现积极情绪"
            else:
                delta_desc = "用户当前情绪延续已有倾向"

        return {
            "baseline": profile,
            "instant": instant_emotion,
            "delta_desc": delta_desc
        }
