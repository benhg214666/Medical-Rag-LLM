"""clean_text 的行為測試。

重點不只是「有沒有清乾淨」，更重要的是「有沒有動到不該動的內容」。
醫療文件中的數值、單位與符號一旦被改寫，可能造成臨床誤解，
因此保留性測試（preservation tests）的份量刻意大於清理測試。
"""

from app.ingestion.cleaner import clean_text


class TestNewlineNormalization:
    """換行統一。"""

    def test_crlf_becomes_lf(self) -> None:
        assert clean_text("第一行\r\n第二行") == "第一行\n第二行"

    def test_lone_cr_becomes_lf(self) -> None:
        assert clean_text("第一行\r第二行") == "第一行\n第二行"

    def test_mixed_line_endings(self) -> None:
        result = clean_text("A\r\nB\rC\nD")
        assert result == "A\nB\nC\nD"
        assert "\r" not in result


class TestWhitespaceCompression:
    """空格與 tab 的壓縮。"""

    def test_multiple_spaces_become_one(self) -> None:
        assert clean_text("血壓     正常") == "血壓 正常"

    def test_tab_becomes_space(self) -> None:
        assert clean_text("項目\t數值") == "項目 數值"

    def test_mixed_space_and_tab(self) -> None:
        assert clean_text("A \t  \t B") == "A B"

    def test_full_width_space_compressed(self) -> None:
        assert clean_text("中文　　空格") == "中文 空格"

    def test_line_leading_and_trailing_stripped(self) -> None:
        assert clean_text("   前後空白   \n   第二行   ") == "前後空白\n第二行"


class TestBlankLineHandling:
    """空白行最多保留一個。"""

    def test_many_blank_lines_reduced_to_one(self) -> None:
        assert clean_text("A\n\n\n\n\n\nB") == "A\n\nB"

    def test_single_blank_line_preserved(self) -> None:
        # 空白行代表段落分隔，是切塊時的重要邊界，不可完全移除
        assert clean_text("段落一\n\n段落二") == "段落一\n\n段落二"

    def test_leading_and_trailing_blank_lines_removed(self) -> None:
        assert clean_text("\n\n\n內容\n\n\n") == "內容"


class TestControlCharacterRemoval:
    """控制字元移除。"""

    def test_null_byte_removed(self) -> None:
        result = clean_text("正常\x00文字")
        assert "\x00" not in result
        assert result == "正常文字"

    def test_other_control_chars_removed(self) -> None:
        result = clean_text("A\x01\x02\x07B")
        assert result == "AB"

    def test_newline_is_not_removed(self) -> None:
        # \n 雖然也是控制字元，但承載段落結構，必須保留
        assert "\n" in clean_text("A\nB")


class TestMedicalContentPreservation:
    """醫療內容保留 —— 這組測試最關鍵。"""

    def test_chinese_preserved(self) -> None:
        text = "病患主訴胸悶、呼吸急促，已持續三天。"
        assert clean_text(text) == text

    def test_english_preserved(self) -> None:
        text = "Patient reports chest pain and dyspnea."
        assert clean_text(text) == text

    def test_numbers_and_decimals_preserved(self) -> None:
        # 0.5 與 05 相差十倍，小數點絕不可遺失
        text = "劑量 0.5 mg，體溫 36.8 度"
        assert clean_text(text) == text

    def test_medical_symbols_preserved(self) -> None:
        text = "血氧 98% / 血壓 120/80 mmHg (正常) - 追蹤中 + 續用藥 : 每日一次；"
        assert clean_text(text) == text

    def test_units_preserved(self) -> None:
        text = "100 mg/dL, 5 mL, 37.2°C, 60 bpm"
        assert clean_text(text) == text

    def test_dosage_expression_not_rewritten(self) -> None:
        text = "Aspirin 100mg PO QD x 7 days"
        assert clean_text(text) == text

    def test_ratio_and_range_preserved(self) -> None:
        text = "WBC 4,500-11,000 /µL；比例 1:2"
        assert clean_text(text) == text

    def test_no_summarization(self) -> None:
        # 清理絕不縮短實質內容，只壓縮空白
        text = "第一項發現。第二項發現。第三項發現。"
        assert clean_text(text) == text


class TestEmptyInput:
    """空輸入的一致行為：一律回傳空字串。"""

    def test_empty_string(self) -> None:
        assert clean_text("") == ""

    def test_only_spaces(self) -> None:
        assert clean_text("     ") == ""

    def test_only_newlines(self) -> None:
        assert clean_text("\n\n\n") == ""

    def test_only_control_chars(self) -> None:
        assert clean_text("\x00\x01\x02") == ""

    def test_mixed_whitespace_only(self) -> None:
        assert clean_text("  \t \n\r\n  \t  ") == ""


class TestIdempotency:
    """清理兩次的結果應與清理一次相同（冪等）。"""

    def test_clean_twice_equals_clean_once(self) -> None:
        raw = "  A  \r\n\r\n\r\n\tB\x00  \n  "
        once = clean_text(raw)
        assert clean_text(once) == once
