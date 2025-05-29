import os
from pypdf import PdfReader
import re
import tiktoken
import jieba
from typing import List, Dict, Any

class AgenticRAG:
    """

    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    
    def load_local_document(self, max_page: int = 1000) -> str:
        """
        从用户本地加载PDF文档，并返回其文本内容。
        支持以中文为主、英文为辅的文档，优化中文分词和编码处理。

        参数:
            file_path: 本地PDF文件绝对路径路径，比如"D:\my_file\document.pdf"
            max_page: 最大页面数限制，默认为1000页

        返回:
            提取的文档文本内容（字符串）
        """
        # 检查文件是否存在
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件 {self.file_path} 不存在，请检查路径！")

        print(f"正在从 {self.file_path} 加载文档...")

        try:
            # 使用 PdfReader 读取本地PDF文件
            pdf_reader = PdfReader(self.file_path)
            full_text = ""

            # 限制最大页面数，防止处理过大文档
            for i, page in enumerate(pdf_reader.pages):
                if i >= max_page:
                    break
                # 提取页面文本，确保正确处理中文编码
                page_text = page.extract_text() or ""
                full_text += page_text + "\n"

            # 清理文本，去除多余空行和空格
            full_text = re.sub(r'\u3000+', ' ', full_text)  # 替换全角空格为普通空格
            full_text = re.sub(r'\n\s*\n', '\n', full_text.strip())

            # 计算词数（中文使用jieba分词，英文按单词统计）
            words = jieba.lcut(full_text)  # 使用jieba进行中文分词
            word_count = len(words)

            # 使用tiktoken计算token数（兼容中英文）
            tokenizer = tiktoken.get_encoding("o200k_base")
            token_count = len(tokenizer.encode(full_text))

            print(f"文档加载完成：共 {len(pdf_reader.pages)} 页，约 {word_count} 词，{token_count} tokens")

            # 展示文档前500个字符预览
            print("\n文档预览（前500个字符）：")
            print("-" * 50)
            print(full_text[:500])
            print("-" * 50)

            return full_text

        except Exception as e:
            raise Exception(f"加载文档时出错：{str(e)}")


# 示例：加载本地PDF文件
if __name__ == "__main__":
    agentic_rag = AgenticRAG(file_path="D:\城市设计模力社区\城市设计模力社区建设方案.pdf")
    try:
        document_text = agentic_rag.load_local_document()
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"错误：{e}")