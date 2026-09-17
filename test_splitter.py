import unittest
from splitter import split_text


class TestSplitText(unittest.TestCase):
    """测试 split_text 核心切分逻辑"""

    def test_empty_text(self):
        """空文本返回空列表"""
        self.assertEqual(split_text(""), [])

    def test_short_text(self):
        """文本长度 <= chunk_size 时直接返回"""
        text = "你好世界"
        self.assertEqual(split_text(text, chunk_size=100), [text])

    def test_exact_size(self):
        """文本刚好等于 chunk_size"""
        text = "A" * 50
        self.assertEqual(split_text(text, chunk_size=50), [text])

    def test_basic_split_by_char(self):
        """无分隔符时按字符数硬切"""
        text = "A" * 100
        chunks = split_text(text, chunk_size=30, chunk_overlap=0)
        self.assertEqual(len(chunks), 4)  # 30+30+30+10
        self.assertEqual(chunks[0], "A" * 30)
        self.assertEqual(chunks[-1], "A" * 10)

    def test_split_by_paragraph(self):
        """段落分隔符（\\n\\n）优先级最高"""
        text = "a\n\nb\n\nc"
        chunks = split_text(text, chunk_size=3, chunk_overlap=0)
        self.assertEqual(len(chunks), 3)
        self.assertIn("a", chunks[0])
        self.assertIn("b", chunks[1])

    def test_split_by_sentence(self):
        """句子分隔符（。！？）次优先级"""
        text = "a。b。c。"
        chunks = split_text(text, chunk_size=3, chunk_overlap=0)
        self.assertEqual(len(chunks), 3)

    def test_overlap_keeps_context(self):
        """重叠区域包含上一块尾部内容"""
        text = "AAABBBCCCDDD"
        chunks = split_text(text, chunk_size=6, chunk_overlap=3)
        self.assertEqual(chunks[0], "AAABBB")
        self.assertIn(chunks[0][-3:], chunks[1])  # 上块尾部=下块头部

    def test_zero_overlap(self):
        """chunk_overlap=0 时无重叠"""
        text = "AAAAAAAAAA"
        chunks = split_text(text, chunk_size=4, chunk_overlap=0)
        self.assertEqual(chunks, ["AAAA", "AAAA", "AA"])

    def test_large_overlap(self):
        """重叠大于chunk_size时不会崩溃"""
        text = "ABCDEFGHIJ"
        chunks = split_text(text, chunk_size=5, chunk_overlap=10)
        # 不会报错，能正常返回即可
        self.assertTrue(len(chunks) >= 1)

    def test_chinese_mixed(self):
        """中英文混合文本"""
        text = "Hello世界！" * 20
        chunks = split_text(text, chunk_size=30, chunk_overlap=5)
        for chunk in chunks:
            self.assertGreater(len(chunk), 0)
        # 所有块拼接起来长度 ≈ 原文本 + 重叠部分
        total = sum(len(c) for c in chunks)
        self.assertGreater(total, len(text))

    def test_short_chunk_size(self):
        """chunk_size=1 的极端情况"""
        text = "ABC"
        chunks = split_text(text, chunk_size=1, chunk_overlap=0)
        self.assertEqual(chunks, ["A", "B", "C"])

    def test_chunks_cover_all_text(self):
        """所有块的并集覆盖了原文"""
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
        chunks = split_text(text, chunk_size=30, chunk_overlap=5)
        combined = "".join(chunks)
        for char in text:
            self.assertIn(char, combined)


if __name__ == "__main__":
    unittest.main(verbosity=2)