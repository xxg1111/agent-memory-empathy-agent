import os
# 必须放在所有import最前面，镜像环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


import uuid
import time
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb


class VectorMemory:
    """第三层 情景向量记忆层，增加相似度阈值过滤防幻觉；add_memory返回记忆UUID供聚类备份"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        # 使用项目本地模型文件夹，完全离线，禁止访问外网
        self.embedding_model = SentenceTransformer(
            model_name_or_path="./bge-small-zh-v1.5",
            local_files_only=True
        )
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "dialog_episodic_memory"
        self.collection = self._get_or_create_collection()
        # 修改阈值：降低阈值提高召回率，论文可做调参实验说明
        self.SIM_THRESHOLD = 0.60
        print("✅ 向量记忆层初始化成功")

    def _get_or_create_collection(self):
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception:
            return self.client.create_collection(name=self.collection_name)

    def add_memory(self, user_id: str, content: str, meta: Optional[Dict] = None) -> str:
        if meta is None:
            meta = {}
        mem_id = str(uuid.uuid4())
        now_ts = time.time()
        metadata = {
            "user_id": user_id,
            "timestamp": now_ts,
            **meta
        }
        embedding = self.embedding_model.encode(content, normalize_embeddings=True).tolist()
        self.collection.add(
            ids=[mem_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )
        print(f"[向量记忆保存] user:{user_id} 内容:{content[:30]}...")
        return mem_id

    def search_memory(self, user_id: str, query: str, top_k: int = 4):
        query_emb = self.embedding_model.encode(query, normalize_embeddings=True).tolist()
        res = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where={"user_id": user_id},
            include=["documents", "distances"]
        )
        if not res or not res.get("documents") or len(res["documents"][0]) == 0:
            return []

        docs = res["documents"][0]
        distances = res["distances"][0]
        valid_docs = []
        for doc, dist in zip(docs, distances):
            # chroma distance = 1‑相似度
            sim = 1.0 - dist
            if sim >= self.SIM_THRESHOLD:
                valid_docs.append(doc)
        return valid_docs

    # 给empathy事件聚类调用，返回numpy数组向量
    def get_embedding(self, text: str):
        return self.embedding_model.encode(text, normalize_embeddings=True)
