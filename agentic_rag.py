import os
import re
from pathlib import Path
from typing import List, Dict, Any
import logging
import json
import asyncio

from pypdf import PdfReader
from pypdf.errors import PdfReadError
import tiktoken
import jieba

from llm import RouterLLM, ReasoningLLM, VerificationLLM

logger = logging.getLogger(__name__)

class AgenticRAG:
    """基于 Agentic RAG 方法的文档处理类，用于加载和处理 PDF 文档。"""

    def __init__(
            self,
            file_path: str,
            user_question: str,
            router_llm: RouterLLM = RouterLLM(),
            min_tokens: int = 500,
            max_chunks: int = 20,
            fine_split: int = 3
    ):
        self.file_path = str(Path(file_path))
        self.user_question = user_question
        self.document_text = None
        self.tokenizer = tiktoken.get_encoding("o200k_base")
        self.min_tokens = min_tokens
        self.max_chunks = max_chunks
        self.fine_split = fine_split
        self.router_llm = router_llm
        self.chunks = []

    def load_local_document(self, max_page: int = 1000) -> str:
        """从本地加载 PDF 文档，返回文本内容。支持中文文档。"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件 {self.file_path} 不存在，请检查路径！")
        logger.info(f"正在从 {self.file_path} 加载文档...")
        try:
            pdf_reader = PdfReader(self.file_path)
            full_text = ""
            for i, page in enumerate(pdf_reader.pages):
                if i >= max_page:
                    break
                page_text = page.extract_text() or ""
                full_text += page_text + "\n"
            full_text = re.sub(r'\u3000+', ' ', full_text)
            full_text = re.sub(r'\n\s*\n', '\n', full_text.strip())
            words = jieba.lcut(full_text)
            word_count = len(words)
            token_count = len(self.tokenizer.encode(full_text))
            logger.info(f"文档加载完成：共 {len(pdf_reader.pages)} 页，约 {word_count} 词，{token_count} tokens")
            self.document_text = full_text
            return full_text
        except PdfReadError as e:
            raise Exception(f"PDF 解析失败：{str(e)}")
        except Exception as e:
            raise Exception(f"加载文档时出错：{str(e)}")

    def split_into_chunks(self, min_tokens: int = None) -> List[Dict[str, Any]]:
        """将文本分成最多 max_chunks 个块，尊重中文句子边界。"""
        if not self.document_text:
            return []
        min_tokens = min_tokens or self.min_tokens
        sentences = [s.strip() for s in re.split(r'(?<=[。！？]|\n)', self.document_text)
                     if s.strip() and len(s.strip()) > 2]

        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            if (current_chunk_tokens + sentence_tokens > min_tokens * 2) and current_chunk_tokens >= min_tokens:
                chunk_text = "".join(current_chunk_sentences)
                chunks.append({"id": len(chunks), "text": chunk_text})
                current_chunk_sentences = [sentence]
                current_chunk_tokens = sentence_tokens
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens

        if current_chunk_sentences:
            chunk_text = "".join(current_chunk_sentences)
            chunks.append({"id": len(chunks), "text": chunk_text})

        if len(chunks) > self.max_chunks:
            all_text = "".join(chunk["text"] for chunk in chunks)
            sentences = [s.strip() for s in re.split(r'(?<=[。！？]|\n)', all_text)
                         if s.strip() and len(s.strip()) > 2]
            sentences_per_chunk = len(sentences) // self.max_chunks + (1 if len(sentences) % self.max_chunks else 0)
            chunks = []
            for i in range(0, len(sentences), sentences_per_chunk):
                chunk_sentences = sentences[i:i + sentences_per_chunk]
                chunk_text = "".join(chunk_sentences)
                chunks.append({"id": len(chunks), "text": chunk_text})

        logger.info(f"文档切分块数合计：{len(chunks)}")
        for i, chunk in enumerate(chunks):
            token_count = len(self.tokenizer.encode(chunk["text"]))
            logger.debug(f"Chunk {i}: {token_count} tokens")
        self.chunks = chunks
        return chunks

    async def coarse_filtration(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """粗滤机制，评估每个块的相关性。"""
        logger.info("\n==== 粗滤阶段 ====")
        logger.info(f"正在评估 {len(chunks)} 个文本块的相关性")

        chunk_messages = []
        for chunk in chunks:
            message = f"问题: {self.user_question}\n\n文本块:\n块 {chunk['id']}:\n{chunk['text']}\n\n"
            message += "请评估该块是否包含回答问题的信息，返回严格的 JSON 格式：\n"
            message += '{"is_relevant": true, "relevance": 0.8, "reasoning": "文本包含关键信息"}'
            message += "\n要求：\n- is_relevant 是布尔值（true/false，无引号）。\n- relevance 是 0.0 到 1.0 的浮点数。\n- reasoning 是双引号包裹的字符串。\n- 确保 JSON 格式合法。"
            chunk_messages.append((chunk['id'], message))

        semaphore = asyncio.Semaphore(5)
        async def limited_chat_completion(msg):
            async with semaphore:
                return await self.router_llm.chat_completion(msg, filtration_stage=0)

        tasks = [limited_chat_completion(message) for _, message in chunk_messages]
        responses = await asyncio.gather(*tasks)

        selected_ids = []
        scratchpad = {}
        for (chunk_id, _), response in zip(chunk_messages, responses):
            try:
                result = json.loads(response)
                is_relevant = result.get("is_relevant", False)
                relevance = float(result.get("relevance", 0.0))
                reasoning = result.get("reasoning", "未提供推理")
                if not isinstance(is_relevant, bool):
                    is_relevant = str(is_relevant).lower() == "true"
                if not 0.0 <= relevance <= 1.0:
                    relevance = 0.0
                    reasoning = f"relevance 超出范围: {relevance}, {reasoning}"
                scratchpad[chunk_id] = {"is_relevant": is_relevant, "relevance": relevance, "reasoning": reasoning}
                if is_relevant:
                    selected_ids.append(chunk_id)
            except json.JSONDecodeError as e:
                logger.warning(f"警告：块 {chunk_id} 的响应无法解析为 JSON: {response}, 错误: {str(e)}")
                scratchpad[chunk_id] = {"is_relevant": False, "relevance": 0.0, "reasoning": f"解析错误: {response}"}

        logger.info(f"选中的块: {', '.join(str(id) for id in selected_ids)}")
        logger.debug(f"Scratchpad 记录: {json.dumps(scratchpad, indent=2, ensure_ascii=False)}")
        return {"selected_ids": selected_ids, "scratchpad": scratchpad}

    async def fine_filtration(self, scratchpad: Dict[str, Any], max_selected_chunks: int = 3) -> Dict:
        """精滤机制，进一步筛选子块。"""
        logger.info("\n==== 精滤阶段 ====")
        sorted_chunks = sorted(scratchpad.items(), key=lambda x: x[1]["relevance"], reverse=True)[:max_selected_chunks]
        selected_chunk_ids = [chunk_id for chunk_id, _ in sorted_chunks]
        logger.info(f"选中的块 ID: {selected_chunk_ids}")

        sub_chunks = []
        for chunk_id in selected_chunk_ids:
            chunk_text = next(chunk["text"] for chunk in self.chunks if chunk["id"] == chunk_id)
            self.document_text = chunk_text
            fine_chunks = self.split_into_chunks(min_tokens=100)
            sub_chunks.extend(fine_chunks[:self.fine_split])

        messages = []
        for sub_chunk in sub_chunks:
            message = f"问题: {self.user_question}\n\n子块:\n{sub_chunk['text']}\n\n粗滤上下文:\n{json.dumps(scratchpad, ensure_ascii=False)}\n\n"
            message += "评估该子块是否直接包含答案，返回严格的 JSON 格式：\n{'is_selected': true, 'reasoning': '推理过程'}"
            message += "\n要求：\n- is_selected 是布尔值（true/false，无引号）。\n- reasoning 是双引号包裹的字符串。\n- 确保 JSON 格式合法。"
            messages.append((sub_chunk, message))

        semaphore = asyncio.Semaphore(5)
        async def limited_chat_completion(msg):
            async with semaphore:
                return await self.router_llm.chat_completion(msg, filtration_stage=1)

        tasks = [limited_chat_completion(msg) for _, msg in messages]
        responses = await asyncio.gather(*tasks)

        selected_sub_chunks = []
        fine_scratchpad = []
        for (sub_chunk, _), response in zip(messages, responses):
            try:
                result = json.loads(response)
                is_selected = result.get("is_selected", False)
                reasoning = result.get("reasoning", "未提供推理")
                if isinstance(is_selected, str):
                    is_selected = is_selected.lower() == "true"
                if is_selected:
                    selected_sub_chunks.append(sub_chunk["text"])
                    fine_scratchpad.append(reasoning)
            except json.JSONDecodeError:
                logger.warning(f"警告：子块 {sub_chunk['id']} 的响应无法解析为 JSON: {response}")
                fine_scratchpad.append(f"解析错误: {response}")

        logger.info(f"选中的子块数: {len(selected_sub_chunks)}")
        return {"selected_sub_chunks": selected_sub_chunks, "scratchpad": fine_scratchpad}

    def generate_answer(self, question: str, paragraphs: List[Dict]) -> Dict:
        """基于筛选的子块生成答案。"""
        logger.info("\n=== 生成答案阶段 ===")
        reasoning_llm = ReasoningLLM()
        sorted_paragraphs = sorted(paragraphs, key=lambda x: x['id'])
        context = "\n".join([f"文本块 {p['id']}:\n{p['text']}" for p in sorted_paragraphs])
        message = f"问题: {question}\n文档块集合:\n{context}\n请根据文档内容回答问题。"
        try:
            answer = reasoning_llm.chat_completion(message)
            logger.info(f"生成答案: {answer}")
            return {"answer": answer, "source_chunks": [p['id'] for p in sorted_paragraphs], "status": "success"}
        except Exception as e:
            logger.error(f"生成答案时出错: {str(e)}")
            return {"answer": "", "source_chunks": [], "status": f"error: {str(e)}"}

    def verify_answer(self, question: str, answer: Dict) -> bool:
        """验证答案准确性。"""
        logger.info("\n=== 验证答案阶段 ===")
        verification_llm = VerificationLLM()
        if not answer.get("answer") or answer.get("status") != "success":
            logger.warning("验证失败：答案无效或生成失败")
            return False
        answer_text = answer["answer"]
        result = verification_llm.chat_completion(question, answer_text)
        logger.debug(f"验证结果: {result}")
        try:
            verification_result = json.loads(result)
            is_correct = verification_result.get("is_correct", False)
            logger.info(f"验证推理: {verification_result.get('reasoning', '未提供推理')}")
            return bool(is_correct)
        except json.JSONDecodeError as e:
            logger.warning(f"验证结果解析失败: {result}, 错误: {str(e)}")
            return False

if __name__ == "__main__":
    pass