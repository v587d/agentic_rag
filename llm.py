import os
from dotenv import load_dotenv
import asyncio
import logging

from openai import (
    OpenAI,
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    APITimeoutError
)
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log
)

# 配置日志
logger = logging.getLogger(__name__)

# 定义通用重试装饰器
def async_retry_on_timeout():
    return retry(
        retry=retry_if_exception_type(
            (APITimeoutError, APIConnectionError, asyncio.TimeoutError)
        ),
        stop=stop_after_attempt(3),  # 最多重试3次
        wait=wait_random_exponential(multiplier=1, max=30),  # 随机抖动 + 指数退避
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
        retry_error_callback=lambda retry_state: None,
    )

class RouterLLM:
    def __init__(
            self,
            max_tokens: int = 1024,
            temperature: float = 0.5,

    ) -> None:
        """
        初始化 LLM 类，加载 OpenAI API Key、 Base Url并设置异步客户端。

        :param max_tokens: 最大生成 token 数，默认为 1024
        :param temperature: 采样温度，默认为 0.5
        :return None
        """
        # 加载 .env 文件中的环境变量
        load_dotenv()

        # 从环境变量中获取模型名称
        model = os.getenv("ROUTER_MODEL_NAME")
        if not model:
            raise ValueError("ROUTER_MODEL_NAME 环境变量未设置，请在 .env 文件中设置你的 ROUTER_MODEL_NAME")

        # 从环境变量中获取base_url
        base_url = os.getenv("ROUTER_MODEL_BASE_URL")
        if not base_url:
            raise ValueError("ROUTER_MODEL_BASE_URL 环境变量未设置，请在 .env 文件中设置你的 ROUTER_MODEL_BASE_URL")

        # 从环境变量中获取 API Key
        api_key = os.getenv("ROUTER_MODEL_API_KEY")
        if not api_key:
            raise ValueError("ROUTER_MODEL_API_KEY 环境变量未设置，请在 .env 文件中设置你的 ROUTER_MODEL_API_KEY")

        # 初始化 OpenAI 客户端
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @async_retry_on_timeout()
    async def chat_completion(
            self,
            message: str,
            filtration_stage: int = 0,
    ) -> str:
        """
        使用 OpenAI Chat Completion API 发送用户消息并获取回复。

        :param message: 用户输入的消息
        :param filtration_stage: filtration_stage=0(粗滤) filtration_stage=1(精滤)
        :return 模型的回复
        """
        system_prompt = ""
        if filtration_stage == 0:
            system_prompt = f"你是一个文档导航助手。你对各行各业，特别是法律、医疗、金融、财税、编程等专业领域都有基本认知。你的任务是：\n"
            system_prompt += "1. 确定文本块是否可能包含回答用户问题的信息。\n"
            system_prompt += "2. 在 scratchpad 中记录你的推理过程以供后续参考。\n"
            system_prompt += "3. 你挑选的文本块必须直接或间接与用户问题相关，避免选择完全不相关文本块。\n\n"
            system_prompt += "首先仔细思考回答问题需要什么信息，然后评估文本块。"
        elif filtration_stage == 1:
            system_prompt = "你是一个文档分析专家，专职于为解答用户问题查找任何可参考的文档片段。\n"
            system_prompt += "你的目标是：\n"
            system_prompt += "1. 严格判断子块是否直接提供问题的准确答案或关键事实。\n"
            system_prompt += "2. 仅选择包含完整、明确回答的子块，排除任何部分相关或间接关联的内容。\n"
            system_prompt += "3. 记录详细推理，解释为何子块被选中或排除。\n\n"
            system_prompt += "操作规则：\n"
            system_prompt += "- 参考粗滤 scratchpad 提供的大致上下文，但决策完全基于子块自身内容。\n"
            system_prompt += "- 答案必须具体且直接，例如数字、日期或明确陈述；模糊或推测性信息不予通过。\n"
            system_prompt += "- 优先考虑子块与问题匹配的精确度，而非数量。\n"
            system_prompt += "开始时，分析问题所需的具体信息，然后逐一评估子块是否满足要求。"
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    ChatCompletionSystemMessageParam(role="system", content=system_prompt),
                    ChatCompletionUserMessageParam(role="user", content=message)
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=30
            )
            return response.choices[0].message.content.strip()
        except Exception as error:
            return f"错误: {str(error)}"

# 示例使用
if __name__ == "__main__":
    pass