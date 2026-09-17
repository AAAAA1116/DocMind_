import os


def load_document(file_path: str) -> str:
    """
    加载文档并提取文本内容，支持 PDF、Word、TXT、Markdown 格式。

    Args:
        file_path: 文档路径

    Returns:
        提取的文本内容

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _load_pdf(file_path)
    elif ext == ".docx":
        return _load_docx(file_path)
    elif ext in (".txt", ".md"):
        return _load_text(file_path)
    else:
        raise ValueError(f"不支持的文件格式：{ext}，仅支持 .pdf / .docx / .txt / .md")


def _load_pdf(file_path: str) -> str:
    """使用 pypdf 解析 PDF 文件"""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("请安装 pypdf：pip install pypdf")

    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def _load_docx(file_path: str) -> str:
    """使用 python-docx 解析 Word 文件"""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("请安装 python-docx：pip install python-docx")

    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paragraphs)


def _load_text(file_path: str) -> str:
    """直接读取纯文本文件（TXT / MD）"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()