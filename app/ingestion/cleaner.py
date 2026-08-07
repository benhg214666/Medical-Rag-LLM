"""文字清理（cleaning）。

為什麼需要清理：從 PDF、DOCX 抽取出來的文字通常帶有大量排版雜訊
（不一致的換行、成串的空格、控制字元）。這些雜訊會稀釋後續 embedding 的語意品質，
也會讓切塊邊界判斷失準。

清理的界線非常重要：本模組只做「格式正規化」，絕不改寫內容。
不摘要、不更正藥名、不調整劑量或數值 —— 醫療文件中任何一個字元的改動都可能造成臨床誤解。
例如 "0.5 mg" 與 "05 mg" 相差十倍，"BP 120/80" 的斜線不能當成雜訊移除。
"""

import re
import unicodedata

# 允許保留的控制字元：只留換行。tab 會在後續被轉為空格。
_ALLOWED_CONTROL_CHARS = {"\n"}

# 連續空格或 tab（不含換行）壓縮為單一空格
_HORIZONTAL_WHITESPACE = re.compile(r"[ \t　]+")

# 三個以上的連續換行壓縮為兩個（即最多保留一個空白行）
_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")


def _remove_control_characters(text: str) -> str:
    """移除 null byte 與其他不必要的控制字元。

    使用 Unicode 分類 Cc（Other, control）判斷，比手動列舉可靠。
    換行字元會被保留，因為它承載了段落結構資訊。
    tab 不在此處移除，而是交由空白壓縮階段轉成空格。
    """
    return "".join(
        char
        for char in text
        if char in _ALLOWED_CONTROL_CHARS
        or char == "\t"
        or unicodedata.category(char) != "Cc"
    )


def clean_text(text: str) -> str:
    """清理單一文件單位的文字內容。

    處理順序（順序有意義，不可任意調換）：
      1. 換行統一為 \\n（先處理 \\r\\n 再處理單獨的 \\r）。
      2. 移除 null byte 與控制字元。
      3. 壓縮連續的空格與 tab。
      4. 去除每行頭尾空白。
      5. 過多空白行最多保留一個。
      6. 去除整份文字頭尾的空白。

    醫療符號如 % / - + . : ( ) 以及中文、英文、數字、單位一律原樣保留。

    Args:
        text: 待清理的原始文字。

    Returns:
        清理後的文字。輸入為空字串或僅含空白時，一律回傳空字串 ""。
    """
    if not text:
        return ""

    # 1. 換行統一
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. 移除控制字元（保留 \n 與 \t）
    normalized = _remove_control_characters(normalized)

    # 3. 壓縮水平空白（半形空格、tab、全形空格）
    normalized = _HORIZONTAL_WHITESPACE.sub(" ", normalized)

    # 4. 去除每行頭尾空白
    normalized = "\n".join(line.strip() for line in normalized.split("\n"))

    # 5. 過多空白行最多保留一個
    normalized = _EXCESSIVE_NEWLINES.sub("\n\n", normalized)

    # 6. 去除整份文字的頭尾空白
    return normalized.strip()
