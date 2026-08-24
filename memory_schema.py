from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class HardFact:
    """
    硬事实键值对结构定义（架构规范预留）
    当前 validator 以字典形式交互，此类留作后续类型强化
    """
    key: str
    value: str
    updated_at: Optional[datetime] = None

@dataclass
class EmotionProfile:
    """
    用户情绪统计画像
    与 SQLite 中 user_emotion 表字段完全对齐
    """
    joy: int = 0
    sadness: int = 0
    anger: int = 0
    fear: int = 0
