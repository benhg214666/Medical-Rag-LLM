"""集中式 logging 設定。

只在應用程式啟動時呼叫一次 setup_logging()，
其他模組一律使用 logging.getLogger(__name__) 取得 logger，不再各自設定。
"""

import logging
from typing import Optional

from app.core.config import settings

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_level: Optional[str] = None) -> None:
    """設定 root logger 的層級與輸出格式。

    Args:
        log_level: 日誌層級字串（例如 "DEBUG"、"INFO"）。
            未提供時採用設定檔中的 LOG_LEVEL。
    """
    level_name: str = (log_level or settings.log_level).upper()
    level: int = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        force=True,
    )
