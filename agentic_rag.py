import os
import re
from pathlib import Path
from typing import List, Dict, Any

from pypdf import PdfReader
import tiktoken
import jieba

from llm import RouterLLM

class AgenticRAG:
    """
    基于Agentic RAG方法的文档处理类，用于加载和处理PDF文档。
    """

    def __init__(
            self,
            file_path: str,
            user_question: str,
            router_llm: RouterLLM = RouterLLM()
    ):
        self.file_path = str(Path(file_path))
        self.user_question = user_question
        self.document_text = None
        self.tokenizer = tiktoken.get_encoding("o200k_base")

    def load_local_document(self, max_page: int = 1000) -> str:
        """
        从用户本地加载PDF文档，并返回其文本内容。
        支持以中文为主、英文为辅的文档，优化中文分词和编码处理。
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件 {self.file_path} 不存在，请检查路径！")
        print(f"正在从 {self.file_path} 加载文档...")
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
            token_count = len(self.tokenizer.encode(full_text))  # 使用类属性 tokenizer
            print(f"文档加载完成：共 {len(pdf_reader.pages)} 页，约 {word_count} 词，{token_count} tokens")
            self.document_text = full_text
            return full_text
        except Exception as e:
            raise Exception(f"加载文档时出错：{str(e)}")

    def split_into_chunks(self, min_tokens: int = 500) -> List[Dict[str, Any]]:
        """
        将文本分成最多20个分块，尊重句子边界，确保每个分块尽量达到min_tokens。
        如果超过20个分块，重新调整为正好20个分块。
        """
        if not self.document_text:
            return []
        # 中文token边界
        # sentences = [s for s in jieba.cut(self.document_text, cut_all=False) if s.strip()]
        # 中文句边界
        sentences = [s.strip() for s in re.split(r'(?<=[。！？]|\n)', text)
                     if s.strip() and len(s.strip()) > 2]

        chunks = []
        current_chunk_sentences = []
        current_chunk_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))
            if (current_chunk_tokens + sentence_tokens > min_tokens * 2) and current_chunk_tokens >= min_tokens:
                chunk_text = "".join(current_chunk_sentences)
                chunks.append({
                    "id": len(chunks),
                    "text": chunk_text
                })
                current_chunk_sentences = [sentence]
                current_chunk_tokens = sentence_tokens
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens

        if current_chunk_sentences:
            chunk_text = "".join(current_chunk_sentences)
            chunks.append({
                "id": len(chunks),
                "text": chunk_text
            })

        if len(chunks) > 20:
            all_text = "".join(chunk["text"] for chunk in chunks)
            sentences = [s for s in jieba.cut(all_text, cut_all=False) if s.strip()]
            sentences_per_chunk = len(sentences) // 20 + (1 if len(sentences) % 20 > 0 else 0)
            chunks = []
            for i in range(0, len(sentences), sentences_per_chunk):
                chunk_sentences = sentences[i:i + sentences_per_chunk]
                chunk_text = "".join(chunk_sentences)
                chunks.append({
                    "id": len(chunks),
                    "text": chunk_text
                })

        print(f"文档切分块数合计： {len(chunks)} ")
        for i, chunk in enumerate(chunks):
            token_count = len(self.tokenizer.encode(chunk["text"]))
            print(f"Chunk {i}: {token_count} tokens")

        return chunks

    async def coarse_filtration(
            self,
            chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        粗滤机制。
        根据用户的问题(self.user_question)，使用异步请求独立评估(self.router_llm)每个文档块（chunk）是否与用户问题相关。
        每个文档块都要维护一个 scratchpad字典，记录了文档块是否相关（True/False），相关性分数（relevance, 0-1），以及原因（推理过程文字描述）。

        :param chunks: 要评估的文档块列表
        :return: 字典，包含选中的块 ID 和 scratchpad字典
        """
        print("\n==== 粗滤阶段 ====")
        print(f"正在评估 {len(chunks)} 个文本块的相关性")

        # 初始化 RouterLLM
        llm = self.router_llm

        # 1. 为每个 chunk 构建用户消息
        chunk_messages = []
        for chunk in chunks:
            message = f"问题: {self.user_question}\n\n"
            message += f"文本块:\n块 {chunk['id']}:\n{chunk['text']}\n\n"
            message += "请评估该块是否包含回答问题的信息，并记录你的推理过程。返回一个 JSON 对象，格式如下：\n"
            message += '{"is_relevant": true/false, "relevance": 0.0-1.0, "reasoning": "你的推理过程"}\n'
            message += "其中：\n"
            message += "- is_relevant 是布尔值（true 或 false），表示该块是否相关；\n"
            message += "- relevance 是一个浮点数（0.0 到 1.0），表示相关性程度，1.0 表示强相关，0.0 表示不相关；\n"
            message += "- relevance 是一个浮点数（0.0 到 1.0），表示相关性程度：\n"
            message += "  - 1.0 表示强相关（文本块直接包含问题的明确答案或核心信息）；\n"
            message += "  - 0.5 表示中相关（文本块可能提供部分相关信息，但不够完整）；\n"
            message += "  - 0.0 表示不相关（文本块与问题完全无关）；\n"
            message += "  - 根据信息匹配度在 0.0 和 1.0 之间打分，例如 0.7 表示部分相关但不够全面；\n"
            message += "- reasoning 是字符串，描述你的推理过程。\n"
            message += "请确保 is_relevant 是布尔值（true 或 false，不用引号），relevance 是浮点数（0.0 到 1.0），reasoning 是字符串。"
            chunk_messages.append((chunk['id'], message))

        # 2. 异步评估所有 chunk
        tasks = [llm.chat_completion(message, filtration_stage=0) for _, message in chunk_messages]
        responses = await asyncio.gather(*tasks)

        # 3. 解析结果并更新 scratchpad 和 selected_ids
        selected_ids = []
        scratchpad = {}

        for (chunk_id, _), response in zip(chunk_messages, responses):
            try:
                # 解析模型返回的 JSON
                result = json.loads(response)
                is_relevant_raw = result.get("is_relevant")
                relevance_raw = result.get("relevance")
                reasoning = result.get("reasoning", "未提供推理")

                # 转换 is_relevant 为布尔值
                if isinstance(is_relevant_raw, str):
                    is_relevant = is_relevant_raw.lower() == "true"
                elif isinstance(is_relevant_raw, bool):
                    is_relevant = is_relevant_raw
                else:
                    is_relevant = False
                    reasoning = f"无效的 is_relevant 值: {is_relevant_raw}, {reasoning}"

                # 转换 relevance 为浮点数，验证范围
                if isinstance(relevance_raw, (int, float)):
                    relevance = float(relevance_raw)
                    if not 0.0 <= relevance <= 1.0:
                        relevance = 0.0
                        reasoning = f"relevance 超出范围（应为 0.0-1.0）: {relevance_raw}, {reasoning}"
                else:
                    relevance = 0.0
                    reasoning = f"无效的 relevance 值: {relevance_raw}, {reasoning}"

                # 更新 scratchpad
                scratchpad[chunk_id] = {
                    "is_relevant": is_relevant,
                    "relevance": relevance,
                    "reasoning": reasoning
                }

                # 如果块被认为相关，添加到 selected_ids
                if is_relevant:
                    selected_ids.append(chunk_id)

            except json.JSONDecodeError:
                print(f"警告：块 {chunk_id} 的响应无法解析为 JSON: {response}")
                scratchpad[chunk_id] = {
                    "is_relevant": False,
                    "relevance": 0.0,
                    "reasoning": f"无法解析响应: {response}"
                }
            except Exception as e:
                print(f"警告：处理块 {chunk_id} 时发生错误: {str(e)}")
                scratchpad[chunk_id] = {
                    "is_relevant": False,
                    "relevance": 0.0,
                    "reasoning": f"错误: {str(e)}"
                }

        # 4. 显示结果
        print(f"选中的块: {', '.join(str(id) for id in selected_ids)}")
        print(f"Scratchpad 记录: {json.dumps(scratchpad, indent=2, ensure_ascii=False)}")

        # 5. 返回结果
        return {
            "selected_ids": selected_ids,
            "scratchpad": scratchpad
        }

    async def fine_filtration(
            self,
            scratchpad: Dict[str, Any],
            max_selected_chunks: int = 3,
            fine_split: int = 3
    ) -> Dict:
        """
        参考了精滤机制。
        根据粗滤结果的相关性排序，从高至低取 top max_selected_chunks 个文本块（chunk）。
        每个文本块，再调 split_into_chunks 方法切成 fine_split 份子文本块（sub_chunks）。
        每份 sub_chunk 和 scratchpad 一起传给routerllm，异步执行精滤评估，找出所有相关的sub_chunks。
        :param scratchpad: coarse_filtration方法返回结果
        :param max_selected_chunks: 子文本块，默认为3
        :param fine_split: 每份 selected_chunk 切成 fine_split 份， 默认为3
        :return:
            返回字典，{"selected_sub_chunks":"content", "scratchpad": "the reason for selecting this chunk"}
        """
        print("\n==== 精滤阶段 ====")

        # 按相关性排序并选取前 max_selected_chunks 个块
        sorted_chunks = sorted(
            scratchpad.items(),
            key=lambda x: x[1]["relevance"],
            reverse=True
        )[:max_selected_chunks]
        selected_chunk_ids = [chunk_id for chunk_id, _ in sorted_chunks]
        print(f"选中的块 ID: {selected_chunk_ids}")

        # 将每个选中的块分割成 fine_split 个子块
        sub_chunks = []
        for chunk_id in selected_chunk_ids:
            chunk_text = next(chunk["text"] for chunk in self.chunks if chunk["id"] == chunk_id)
            fine_chunks = self.split_into_chunks(chunk_text, min_tokens=100)  # 更小的 token 阈值
            sub_chunks.extend(fine_chunks[:fine_split])  # 限制为 fine_split 个子块
        print(f"生成的总子块数: {len(sub_chunks)}")

        # 为 RouterLLM 构建评估消息
        messages = []
        for sub_chunk in sub_chunks:
            message = f"问题: {self.user_question}\n\n"
            message += f"子块:\n{sub_chunk['text']}\n\n"
            message += f"粗滤上下文 (scratchpad):\n{json.dumps(scratchpad, ensure_ascii=False)}\n\n"
            message += "评估该子块是否直接包含问题的答案或核心信息。\n"
            message += "返回一个 JSON 对象，格式如下：\n"
            message += '{"is_selected": true/false, "reasoning": "你的推理过程"}\n'
            message += "确保 'is_selected' 是布尔值（true 或 false，不用引号），'reasoning' 是字符串。"
            messages.append((sub_chunk, message))

        # 异步评估所有子块
        tasks = [self.router_llm.chat_completion(msg, filtration_stage=1) for _, msg in messages]
        responses = await asyncio.gather(*tasks)

        # 解析响应并收集结果
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
                print(f"警告：子块 {sub_chunk['id']} 的响应无法解析为 JSON: {response}")
                fine_scratchpad.append(f"解析错误: {response}")
            except Exception as e:
                print(f"处理子块 {sub_chunk['id']} 时出错: {str(e)}")
                fine_scratchpad.append(f"错误: {str(e)}")

        print(f"选中的子块数: {len(selected_sub_chunks)}")
        return {
            "selected_sub_chunks": selected_sub_chunks,
            "scratchpad": fine_scratchpad
        }

    def generate_answer(question: str, paragraphs: List[Dict]) -> Dict:
        pass

    def verify_answer(question: str, answer: Dict) -> bool:
        pass

if __name__ == "__main__":
    # abs_path = "D:\城市设计模力社区\城市设计模力社区建设方案.pdf"
    # agent = AgenticRAG(file_path=abs_path)
    # try:
    #     document_text = agent.load_local_document()
    #     chunks = agent.split_into_chunks(min_tokens=500)
    #     for chunk in chunks:
    #         token_count = len(agent.tokenizer.encode(chunk["text"]))
    #         print(f"Chunk {chunk['id']}: {token_count} tokens, 内容: {chunk['text'][:50]}...")
    #         if token_count < 500 and chunk['id'] != len(chunks) - 1:
    #             print(f"警告：分块 {chunk['id']} token 数 {token_count} 小于 min_tokens (500)!")
    # except FileNotFoundError as e:
    #     print(e)
    # except Exception as e:
    #     print(f"错误：{e}")
    pass