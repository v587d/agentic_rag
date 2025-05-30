import os
from dotenv import load_dotenv
import asyncio
import logging

from openai import OpenAI, AsyncOpenAI, APIConnectionError, APITimeoutError
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential, before_sleep_log

load_dotenv()
logger = logging.getLogger(__name__)

def async_retry_on_timeout():
    return retry(
        retry=retry_if_exception_type((APITimeoutError, APIConnectionError, asyncio.TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )

class LLM:
    def __init__(self, model_name: str, base_url: str, api_key: str, temperature: float = 0.5, is_async: bool = False):
        if not model_name:
            raise ValueError("model_name 不能为空")
        if not base_url:
            raise ValueError("base_url 不能为空")
        if not api_key:
            raise ValueError("api_key 不能为空")
        self.model = model_name
        self.temperature = temperature
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key) if is_async else OpenAI(base_url=base_url, api_key=api_key)

    @async_retry_on_timeout()
    async def async_chat_completion(self, message: str, system_prompt: str) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    ChatCompletionSystemMessageParam(role="system", content=system_prompt),
                    ChatCompletionUserMessageParam(role="user", content=message)
                ],
                temperature=self.temperature,
                timeout=30
            )
            return response.choices[0].message.content.strip()
        except Exception as error:
            return f"错误: {str(error)}"

    def sync_chat_completion(self, message: str, system_prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    ChatCompletionSystemMessageParam(role="system", content=system_prompt),
                    ChatCompletionUserMessageParam(role="user", content=message)
                ],
                temperature=self.temperature,
                timeout=30
            )
            return response.choices[0].message.content.strip()
        except Exception as error:
            return f"错误: {str(error)}"

class RouterLLM(LLM):
    def __init__(self):
        model_name = os.getenv("ROUTER_MODEL_NAME")
        base_url = os.getenv("ROUTER_MODEL_BASE_URL")
        api_key = os.getenv("ROUTER_MODEL_API_KEY")
        if not model_name:
            raise ValueError("ROUTER_MODEL_NAME 环境变量未设置，请在 .env 文件中配置")
        if not base_url:
            raise ValueError("ROUTER_MODEL_BASE_URL 环境变量未设置，请在 .env 文件中配置")
        if not api_key:
            raise ValueError("ROUTER_MODEL_API_KEY 环境变量未设置，请在 .env 文件中配置")
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, temperature=0.5, is_async=True)

    async def chat_completion(self, message: str, filtration_stage: int = 0) -> str:
        system_prompt = self.get_system_prompt(filtration_stage)
        return await self.async_chat_completion(message, system_prompt)

    def get_system_prompt(self, filtration_stage: int) -> str:
        if filtration_stage == 0:
            return """
你是一个文档导航助手。你的任务是：
1. 确定文本块是否可能包含回答用户问题的信息。
2. 在 scratchpad 中记录你的推理过程。
3. 返回严格的 JSON 格式：
   {
     "is_relevant": true/false,
     "relevance": 0.0-1.0,
     "reasoning": "你的推理过程"
   }
   - is_relevant 是布尔值（true/false，无引号）。
   - relevance 是 0.0 到 1.0 的浮点数。
   - reasoning 是双引号包裹的字符串。
   - 确保 JSON 格式合法，无多余空格或换行。
"""
        elif filtration_stage == 1:
            return """
你是一个文档分析专家，专职于为解答用户问题查找文档片段。
你的目标是：
1. 严格判断子块是否直接提供问题的准确答案或关键事实。
2. 返回严格的 JSON 格式：
   {
     "is_selected": true/false,
     "reasoning": "你的推理过程"
   }
   - is_selected 是布尔值（true/false，无引号）。
   - reasoning 是双引号包裹的字符串。
   - 确保 JSON 格式合法，无多余空格或换行。
"""
        else:
            return "未知的 filtration_stage"

class ReasoningLLM(LLM):
    def __init__(self):
        model_name = os.getenv("REASONING_MODEL_NAME")
        base_url = os.getenv("REASONING_MODEL_BASE_URL")
        api_key = os.getenv("REASONING_MODEL_API_KEY")
        if not model_name:
            raise ValueError("REASONING_MODEL_NAME 环境变量未设置，请在 .env 文件中配置")
        if not base_url:
            raise ValueError("REASONING_MODEL_BASE_URL 环境变量未设置，请在 .env 文件中配置")
        if not api_key:
            raise ValueError("REASONING_MODEL_API_KEY 环境变量未设置，请在 .env 文件中配置")
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, temperature=0.7, is_async=False)

    def chat_completion(self, message: str) -> str:
        system_prompt = """
你是一个通用文档推理专家，专注于从提供的文档块中提取信息并生成准确、详尽的答案。
你的任务是：
1. 仔细分析用户问题。
2. 根据文档块ID索引，从小到大串联所有文档块，分析前后逻辑关系，了解全局。
3. 基于文档块提供的事实，生成逻辑清晰、结构化的答案。
"""
        return self.sync_chat_completion(message, system_prompt)

class VerificationLLM(LLM):
    def __init__(self):
        model_name = os.getenv("VERIFICATION_MODEL_NAME")
        base_url = os.getenv("VERIFICATION_MODEL_BASE_URL")
        api_key = os.getenv("VERIFICATION_MODEL_API_KEY")
        if not model_name:
            raise ValueError("VERIFICATION_MODEL_NAME 环境变量未设置，请在 .env 文件中配置")
        if not base_url:
            raise ValueError("VERIFICATION_MODEL_BASE_URL 环境变量未设置，请在 .env 文件中配置")
        if not api_key:
            raise ValueError("VERIFICATION_MODEL_API_KEY 环境变量未设置，请在 .env 文件中配置")
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, temperature=0.3, is_async=False)

    def chat_completion(self, message: str, answer: str) -> str:
        system_prompt = """
你是一个通用答案验证助手，任务是验证提供的答案是否准确、完整且与用户问题和文档内容一致。
返回严格的 JSON 格式：
{
  "is_correct": true/false,
  "reasoning": "验证过程的详细说明"
}
- is_correct 是布尔值（true/false，无引号）。
- reasoning 是双引号包裹的字符串。
- 确保 JSON 格式合法，无多余空格或换行。
"""
        user_message = f"问题: {message}\n答案: {answer}\n请验证答案的准确性并返回 JSON 格式的结果。"
        return self.sync_chat_completion(user_message, system_prompt)

if __name__ == "__main__":
    pass