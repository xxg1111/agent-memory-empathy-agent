import sqlite3
from typing import Dict, Optional, List
from datetime import datetime
import numpy as np

class EmpathyFilter:
    """
    路线一：事件向量聚类 + LLM双重判断
    1.时间衰减 DECAY_FACTOR=0.94
    2.瞬时情绪 / 长期情绪基线分离
    3.基线样本门槛 MIN_BASELINE_TOTAL=4.0
    4.同事件降权；不同事件正常权重
    """
    VALID_EMOTIONS = {"joy", "sadness", "anger", "fear"}
    DECAY_FACTOR = 0.94
    MIN_BASELINE_TOTAL = 4.0
    JOY_THRESHOLD = 0.35
    SADNESS_THRESHOLD = 0.35

    def __init__(self, vector_memory, db_path="agent_memory.db", validator=None):
        self.vector_memory = vector_memory
        self.validator = validator
        self.DB_PATH = db_path
        self._init_table()

    def _get_conn(self):
        conn = sqlite3.connect(self.DB_PATH, check_same_thread=False)
        return conn

    def _init_table(self):
        conn = self._get_conn()
        conn.execute('''
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

    def _get_profile(self, user_id:str) -> Dict:
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT joy,sadness,anger,fear FROM user_emotion WHERE user_id=?",(user_id,))
        row = cur.fetchone()
        if row is None:
            profile = {"joy":0.0,"sadness":0.0,"anger":0.0,"fear":0.0}
            cur.execute('''INSERT INTO user_emotion(user_id,joy,sadness,anger,fear) VALUES (?,?,?,?,?)''',
                        (user_id,0.0,0.0,0.0,0.0))
            conn.commit()
        else:
            profile = {"joy":row[0],"sadness":row[1],"anger":row[2],"fear":row[3]}
        conn.close()
        return profile

    def _decay_profile(self, profile:Dict) -> Dict:
        out = {}
        for k in self.VALID_EMOTIONS:
            out[k] = profile[k] * self.DECAY_FACTOR
        return out

    def _save_profile(self, user_id:str, profile:Dict):
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute('''
        UPDATE user_emotion SET joy=?,sadness=?,anger=?,fear=?,last_update_time=? WHERE user_id=?
        ''',(profile["joy"],profile["sadness"],profile["anger"],profile["fear"],datetime.now().timestamp(),user_id))
        conn.commit()
        conn.close()

    def update_emotion(self, user_id: str, user_input: str, instant_emotion: str, retrieved_memories: List, raw_recent_dialog: List, cluster_id:int):
        """
        :param user_id: 用户id
        :param user_input: 当前用户输入文本
        :param instant_emotion: 瞬时情绪 joy/sadness/anger/fear
        :param retrieved_memories: 向量检索返回记忆列表
        :param raw_recent_dialog: 原始对话缓存列表
        :param cluster_id: hdbscan输出簇id，-1代表噪声簇
        :return: 情绪画像dict
        """
        profile = self._get_profile(user_id)
        # 时间衰减旧基线
        profile = self._decay_profile(profile)

        if instant_emotion not in self.VALID_EMOTIONS:
            return {
                "instant_emotion":"neutral",
                "baseline":profile,
                "delta_desc":"暂无足够历史，重点关注本轮当下情绪"
            }

        base_weight = 1.0
        # 同一事件簇，降权0.2；噪声簇保持1.0
        if cluster_id != -1 and self.validator is not None:
            base_weight = 0.2
            print(f"[Empathy] 检测到属于同一事件簇 id={cluster_id},权重降为 {base_weight}")

        delta = 1.0 * base_weight
        profile[instant_emotion] += delta
        # 限幅
        for k in profile:
            profile[k] = min(profile[k],10.0)

        self._save_profile(user_id, profile)

        total_score = sum(profile.values())
        delta_desc = "暂无足够历史，重点关注本轮当下情绪"
        if total_score >= self.MIN_BASELINE_TOTAL:
            if profile["joy"] >= self.JOY_THRESHOLD and profile["sadness"] < 0.8:
                delta_desc = "用户整体心态偏积极"
            elif profile["sadness"] >= self.SADNESS_THRESHOLD and profile["joy"] <0.8:
                delta_desc = "检测到情绪偏移：用户近期长期处于负面状态"
            else:
                delta_desc = "用户情绪波动比较明显"

        return {
            "instant_emotion": instant_emotion,
            "baseline": profile,
            "delta_desc": delta_desc
        }

    def get_emotion_profile(self,user_id:str):
        return self._get_profile(user_id)
