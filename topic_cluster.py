# topic_cluster.py
import numpy as np
import hdbscan
from typing import List, Dict


def run_hdbscan_cluster(item_list: List[Dict], min_cluster_size: int = 3):
    """
    HDBSCAN聚类；先L2归一化，用euclidean距离等价余弦相似度，兼容新版sklearn
    :param item_list: [{"mem_id":str,"emb":np.ndarray}, ...]
    :param min_cluster_size: 构成一个有效话题簇最少消息条数
    :return: dict key:mem_id  value:cluster_id，-1代表孤立噪声点
    """
    if len(item_list) < min_cluster_size:
        return {item["mem_id"]: -1 for item in item_list}

    emb_matrix = np.array([item["emb"] for item in item_list])
    # L2归一化后，欧式距离等价余弦距离
    norm = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    emb_normalized = emb_matrix / norm

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        allow_single_cluster=False
    )
    label_array = clusterer.fit_predict(emb_normalized)

    result_map = {}
    for idx, item in enumerate(item_list):
        result_map[item["mem_id"]] = int(label_array[idx])
    return result_map


def get_latest_cluster_id(all_items: List[Dict], current_mem_uuid: str) -> int:
    for it in all_items:
        if it["mem_id"] == current_mem_uuid:
            cid = it.get("cluster_id", -1)
            return cid if cid is not None else -1
    return -1
