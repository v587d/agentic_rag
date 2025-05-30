import asyncio
import os
import logging
import json
import re
from datetime import datetime

from agentic_rag import AgenticRAG
from llm import LLM

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

def generate_filename(question: str) -> str:
    """生成唯一的 Markdown 文件名，基于时间戳和问题摘要。"""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # 提取问题前 20 个字符，去除非法文件名字符
    summary = re.sub(r'[^\w\s-]', '', question)[:20].strip().replace(' ', '_')
    if not summary:
        summary = 'unnamed'
    return f"qa_logs/{timestamp}_{summary}.md"

def save_qa_record(content: str, filename: str):
    """将问答记录保存为 Markdown 文件。"""
    os.makedirs('qa_logs', exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"问答记录已保存至：{filename}")

async def main():
    """程序主入口，解析用户输入并运行 AgenticRAG 流程，保存问答记录。"""
    print("=== Agentic RAG for Chinese Context v0.1.0 ===")
    print("🚀 欢迎探索超长中文文档问答系统！")
    print("👨‍💻 开发者：v587d - AI & 开源爱好者")
    print("🌐 GitHub: https://github.com/v587d - 欢迎 Star 和贡献！")
    print(f"🕒 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💡 提示：输入问题和文档路径，例如：请回答合同中关于违约责任的规定是什么？文档路径是 D:/docs/contract.pdf")

    logger.info("欢迎使用文档问答系统！")
    logger.info("请输入您的问题和文档的本地绝对路径，例如：")
    logger.info("请回答合同中关于违约责任的规定是什么？文档路径是 D:/docs/contract.pdf")
    logger.info("或：我想知道违约条款，文件在 C:\\documents\\contract.pdf")

    user_input = input("> ").strip()
    if not user_input:
        logger.error("错误：输入不能为空。")
        return

    # 初始化 Markdown 内容
    md_content = f"# 问答记录\n\n**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += f"## 用户输入\n```\n{user_input}\n```\n\n"

    llm = LLM(
        model_name=os.getenv("REASONING_MODEL_NAME"),
        base_url=os.getenv("REASONING_MODEL_BASE_URL"),
        api_key=os.getenv("REASONING_MODEL_API_KEY"),
        temperature=0.3,
        is_async=False
    )
    system_prompt = """
你是一个智能助手，任务是从用户输入中提取以下信息：
1. 用户问题：用户想要查询的具体问题。
2. 文档路径：本地 PDF 文件的绝对路径。

输入可能是自然语言，例如：
- “请回答合同中关于违约责任的规定是什么？文档路径是 D:/docs/contract.pdf”
- “我想知道违约相关条款，文件在 C:\\documents\\contract.pdf”
- “帮我查一下合同的签署日期，PDF 文件是 /home/user/contract.pdf”

请分析输入，返回严格的 JSON 格式：
{
  "question": "提取的问题",
  "doc_path": "提取的文档路径",
  "reasoning": "提取的推理过程"
}
- 所有字段是双引号包裹的字符串。
- 确保 JSON 格式合法，无多余空格或换行。

规则：
- 如果无法提取问题或路径，返回空字符串并在 reasoning 中说明原因。
- 路径应保留原始格式（包括斜杠或反斜杠）。
- 路径可能使用 Windows 的反斜杠（例如 C:\\docs\\file.pdf）或 Unix 的正斜杠（例如 /home/user/file.pdf），请正确识别。
- 如果路径包含空格或特殊字符，保持原样。
- reasoning 字段解释你的提取逻辑。
- 问题应简洁，聚焦核心查询内容。
- 如果输入不明确，尝试推测合理的意图。
"""
    user_message = f"用户输入：{user_input}\n请分析并提取问题和文档路径，返回 JSON 格式。"

    try:
        result = llm.sync_chat_completion(user_message, system_prompt)
        parsed_result = json.loads(result)
        question = parsed_result.get("question", "").strip()
        doc_path = parsed_result.get("doc_path", "").strip()
        reasoning = parsed_result.get("reasoning", "未提供推理")

        logger.info(f"\nLLM 解析结果：")
        logger.info(f"问题：{question}")
        logger.info(f"文档路径：{doc_path}")
        logger.debug(f"解析推理：{reasoning}")

        md_content += "## LLM 解析结果\n"
        md_content += f"- **问题**: {question}\n"
        md_content += f"- **文档路径**: {doc_path}\n"
        md_content += f"- **解析推理**: {reasoning}\n\n"

        if not question:
            logger.error("错误：无法提取有效问题。")
            md_content += "## 错误\n无法提取有效问题。\n"
            save_qa_record(md_content, generate_filename("invalid_question"))
            return
        if not doc_path:
            logger.error("错误：无法提取有效文档路径。")
            md_content += "## 错误\n无法提取有效文档路径。\n"
            save_qa_record(md_content, generate_filename("invalid_path"))
            return

        doc_path = os.path.normpath(doc_path)
        if not os.path.exists(doc_path):
            logger.error(f"错误：文档路径 {doc_path} 不存在，请检查路径是否正确。")
            md_content += f"## 错误\n文档路径 {doc_path} 不存在。\n"
            save_qa_record(md_content, generate_filename(question))
            return
        if not doc_path.lower().endswith('.pdf'):
            logger.error("错误：只支持 PDF 文档，请提供 .pdf 文件。")
            md_content += "## 错误\n只支持 PDF 文档。\n"
            save_qa_record(md_content, generate_filename(question))
            return

    except json.JSONDecodeError as e:
        logger.error(f"错误：LLM 返回无效 JSON：{repr(result)}, 异常：{str(e)}")
        md_content += f"## 错误\nLLM 返回无效 JSON：\n```\n{repr(result)}\n```\n**异常详情**：{str(e)}\n"
        save_qa_record(md_content, generate_filename("json_error"))
        return
    except Exception as e:
        logger.error(f"错误：解析用户输入时出错：{str(e)}")
        md_content += f"## 错误\n解析用户输入时出错：{str(e)}\n"
        save_qa_record(md_content, generate_filename("parse_error"))
        return

    try:
        agent = AgenticRAG(file_path=doc_path, user_question=question)
        agent.load_local_document()
        chunks = agent.split_into_chunks()
        if not chunks:
            logger.error("错误：文档分块失败，文档可能为空。")
            md_content += "## 错误\n文档分块失败，文档可能为空。\n"
            save_qa_record(md_content, generate_filename(question))
            return

        md_content += "## 文档处理\n"
        md_content += f"- **分块数**: {len(chunks)}\n"

        coarse_result = await agent.coarse_filtration(chunks)
        if not coarse_result["selected_ids"]:
            logger.warning("警告：粗滤阶段未找到相关文档块，可能无法回答问题。")
            md_content += "- **粗滤结果**: 未找到相关文档块。\n"
        else:
            md_content += f"- **粗滤结果**: 选中的块 ID: {', '.join(str(chunk_id) for chunk_id in coarse_result['selected_ids'])}\n"
            md_content += f"  - Scratchpad: ```json\n{json.dumps(coarse_result['scratchpad'], indent=2, ensure_ascii=False)}\n```\n"

        fine_result = await agent.fine_filtration(coarse_result["scratchpad"])
        if not fine_result["selected_sub_chunks"]:
            logger.warning("警告：精滤阶段未找到相关子块，可能无法生成准确答案。")
            md_content += "- **精滤结果**: 未找到相关子块。\n"
        else:
            md_content += f"- **精滤结果**: 选中的子块数: {len(fine_result['selected_sub_chunks'])}\n"
            md_content += f"  - Scratchpad: ```\n{json.dumps(fine_result['scratchpad'], indent=2, ensure_ascii=False)}\n```\n"

        paragraphs = [{"id": i, "text": text} for i, text in enumerate(fine_result["selected_sub_chunks"])]
        answer = agent.generate_answer(question, paragraphs)
        if answer["status"] != "success":
            logger.error(f"错误：答案生成失败：{answer['status']}")
            md_content += f"## 错误\n答案生成失败：{answer['status']}\n"
            save_qa_record(md_content, generate_filename(question))
            return

        is_correct = agent.verify_answer(question, answer)
        md_content += "## 最终结果\n"
        md_content += f"- **答案**: \n```\n{answer['answer']}\n```\n"
        md_content += f"- **答案是否正确**: {'是' if is_correct else '否'}\n"
        md_content += f"- **使用的文档块 ID**: {', '.join(str(id) for id in answer['source_chunks'])}\n"

        # 保存 Markdown 文件
        save_qa_record(md_content, generate_filename(question))

        # 控制台输出最终结果
        logger.info(f"\n最终结果：")
        logger.info(f"答案：{answer['answer']}")
        logger.info(f"答案是否正确：{'是' if is_correct else '否'}")

    except Exception as e:
        logger.error(f"处理流程时出错：{str(e)}")
        md_content += f"## 错误\n处理流程时出错：{str(e)}\n"
        save_qa_record(md_content, generate_filename(question))

if __name__ == "__main__":
    asyncio.run(main())