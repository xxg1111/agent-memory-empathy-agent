import sqlite3
import pickle
import numpy as np
from typing import Dict, Optional
from datetime import datetime


class MemoryValidator:
    MAX_FACTS_LIMIT = 100

    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._init_tables()
        print(f"✅ SQLite 存储初始化完成，数据库路径：{self.db_path}")

    def _get_conn(self):
        """获取临时数据库连接，每次调用新建"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_tables(self):
        conn = self._get_conn()
        cursor = conn.cursor()

        # 用户硬事实表（user_id+fact_key唯一约束，支持自动替换更新）
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            fact_key TEXT NOT NULL,
            fact_value TEXT NOT NULL,
            source_text TEXT,
            updated_at TIMESTAMP NOT NULL,
            UNIQUE(user_id, fact_key)
        )
        ''')

        # 事实变更历史表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS fact_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            fact_key TEXT NOT NULL,
            old_value TEXT NOT NULL,
            new_value TEXT NOT NULL,
            source_text TEXT,
            change_time TIMESTAMP
        )
        ''')

        # 用户情绪统计表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_emotion (
            user_id TEXT PRIMARY KEY,
            joy REAL DEFAULT 0.0,
            sadness REAL DEFAULT 0.0,
            anger REAL DEFAULT 0.0,
            fear REAL DEFAULT 0.0,
            last_update_time REAL
        )
        ''')

        # 向量备份表，用于HDBSCAN话题聚类
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory_emb_backup (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            cluster_id INTEGER NULL,
            ts REAL NOT NULL
        )
        ''')

        # 每个簇最后一次情绪更新时间，避免同一话题情绪重复叠加
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cluster_last_update (
            cluster_id INTEGER PRIMARY KEY,
            last_update_ts REAL NOT NULL
        )
        ''')

        conn.commit()
        conn.close()

    def save_fact(self, user_id: str, key: str, value: str, user_input_text: str):
        """
        保存硬事实
        - 如果key已存在：旧值写入历史表，主记录更新
        - 如果key不存在：直接新增
        :param user_id: 用户id
        :param key: 事实key
        :param value: 事实值
        :param user_input_text: 用户原始句子，用于校验居住地
        """
        # ========= 居住地兜底校验 =========
        if key == "居住地":
            live_words = {"住在", "家在", "长期居住", "定居"}
            has_live = any(w in user_input_text for w in live_words)
            if not has_live:
                print(f"[校验拦截] 拒绝写入居住地={value}，原文没有居住相关描述")
                return None

        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()

            # 计数，控制单用户事实上限
            cursor.execute("SELECT COUNT(*) FROM user_facts WHERE user_id = ?", (user_id,))
            count = cursor.fetchone()[0]
            if count >= self.MAX_FACTS_LIMIT:
                print(f"[警告] user:{user_id} 硬事实已经到达上限 {self.MAX_FACTS_LIMIT}")
                return None

            # 查询旧值
            cursor.execute(
                "SELECT fact_value FROM user_facts WHERE user_id = ? AND fact_key = ?",
                (user_id, key)
            )
            row = cursor.fetchone()
            now = datetime.now().isoformat()

            if row is not None:
                old_val = row[0]
                if old_val.strip() == value.strip():
                    # 值没变，直接跳过（幂等，防止LangGraph重放重复调用）
                    return None
                # 写入历史
                cursor.execute('''
                    INSERT INTO fact_history
                    (user_id, fact_key, old_value, new_value, source_text, change_time)
                    VALUES (?,?,?,?,?,?)
                ''', (user_id, key, old_val, value, user_input_text, now))
                # 更新主记录
                cursor.execute('''
                    UPDATE user_facts
                    SET fact_value=?, source_text=?, updated_at=?
                    WHERE user_id=? AND fact_key=?
                ''', (value, user_input_text, now, user_id, key))
                print(f"[事实自动更新] key={key}: {old_val} → {value}")
            else:
                # 新增
                cursor.execute('''
                INSERT INTO user_facts (user_id, fact_key, fact_value, source_text, updated_at)
                VALUES (?,?,?,?,?)
                ''', (user_id, key, value, user_input_text, now))
                print(f"[事实新增保存] key={key} → {value}")

            conn.commit()
            return True
        except sqlite3.OperationalError as e:
            print(f"[DB异常 save_fact] {str(e)}")
            return False
        finally:
            if conn:
                conn.close()

    def get_all_facts(self, user_id: str) -> Dict[str, str]:
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
            SELECT fact_key, fact_value FROM user_facts WHERE user_id = ?
            ''', (user_id,))
            rows = cursor.fetchall()
            return {k: v for k, v in rows}
        except sqlite3.OperationalError as e:
            print(f"[DB异常 get_all_facts] {str(e)}")
            return {}
        finally:
            if conn:
                conn.close()

    def get_all_history(self, user_id: str, limit=5):
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute("""
                SELECT fact_key,old_value,new_value,change_time
                FROM fact_history
                WHERE user_id=?
                ORDER BY change_time DESC
                LIMIT ?
            """, (user_id, limit))
            rows = cur.fetchall()
            return rows
        except sqlite3.OperationalError as e:
            print(f"[DB异常 get_all_history] {str(e)}")
            return []
        finally:
            if conn:
                conn.close()

    def delete_fact(self, user_id: str, key: str):
        conn = None
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute('''
            DELETE FROM user_facts WHERE user_id = ? AND fact_key = ?
            ''', (user_id, key))
            conn.commit()
        except sqlite3.OperationalError as e:
            print(f"[DB异常 delete_fact] {str(e)}")
        finally:
            if conn:
                conn.close()

    # ============ 聚类新增接口 ============
    def save_emb_backup(self, mem_uuid: str, user_id: str, content: str, emb: np.ndarray, ts: float):
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            blob = pickle.dumps(emb)
            cur.execute('''
            INSERT OR REPLACE INTO memory_emb_backup(id,user_id,content,embedding,cluster_id,ts)
            VALUES (?,?,?,?,NULL,?)
            ''', (mem_uuid, user_id, content, blob, ts))
            conn.commit()
        except sqlite3.OperationalError as e:
            print(f"[DB异常 save_emb_backup] {str(e)}")
        finally:
            if conn:
                conn.close()

    def load_user_all_emb(self, user_id: str):
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute('''
            SELECT id,content,embedding,cluster_id,ts FROM memory_emb_backup WHERE user_id=? ORDER BY ts ASC
            ''', (user_id,))
            rows = cur.fetchall()
            out = []
            for mem_id, text, blob, cid, ts in rows:
                emb = pickle.loads(blob)
                out.append({
                    "mem_id": mem_id,
                    "text": text,
                    "emb": emb,
                    "cluster_id": cid,
                    "ts": ts
                })
            return out
        except sqlite3.OperationalError as e:
            print(f"[DB异常 load_user_all_emb] {str(e)}")
            return []
        finally:
            if conn:
                conn.close()

    def update_cluster_id(self, mem_id: str, cluster_id: int):
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute('UPDATE memory_emb_backup SET cluster_id=? WHERE id=?', (cluster_id, mem_id))
            conn.commit()
        except sqlite3.OperationalError as e:
            print(f"[DB异常 update_cluster_id] {str(e)}")
        finally:
            if conn:
                conn.close()

    def get_cluster_last_ts(self, cluster_id: int) -> Optional[float]:
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute('SELECT last_update_ts FROM cluster_last_update WHERE cluster_id=?', (cluster_id,))
            row = cur.fetchone()
            if row:
                return row[0]
            return None
        except sqlite3.OperationalError as e:
            print(f"[DB异常 get_cluster_last_ts] {str(e)}")
            return None
        finally:
            if conn:
                conn.close()

    def set_cluster_last_ts(self, cluster_id: int, ts: float):
        conn = None
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            cur.execute('''
            INSERT OR REPLACE INTO cluster_last_update(cluster_id,last_update_ts)
            VALUES (?,?)
            ''', (cluster_id, ts))
            conn.commit()
        except sqlite3.OperationalError as e:
            print(f"[DB异常 set_cluster_last_ts] {str(e)}")
        finally:
            if conn:
                conn.close()
