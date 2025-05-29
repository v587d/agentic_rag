# Agentic RAG MCP Server

本项目受到 [OpenAI cookbook：Long-Context RAG for Legal Q&A](https://cookbook.openai.com/examples/partners/model_selection_guide/model_selection_guide#3a-use-case-long-context-rag-for-legal-qa) 的启发，尝试实现基于LLM的RAG MCP Server。

- 本项目期望准确回答复杂冗长的文本问题，比如工具书（辞典、百科全书、年鉴等）、法律文档、技术文档、监管框架等，其中准确性、引用和可审计性是关键任务要求。
- 利用百万级tokens上下文窗口来处理大型文档，而**无需任何预处理或向量数据库**。
- 可实现零延迟提取、动态粒度的检索以及细粒度的引用可追溯性。
- 工作流：这种分层导航方法模仿了人类浏览文档的方式，首先关注相关章节，然后关注特定部分，最后只阅读最相关的段落。
  1. 将整个文档加载到上下文窗口
  2. 切分成若干句子块
  3. 询问大模型哪些块可能回答用户问题
  4. 进一步拆分需关注的块
  5. 重复若干次直到找到所有相关块
  6. 生成答案
  7. 验证答案

## LLM

本项目需配置多个大模型，发挥不同模型的特性，相互作用。

- 检索模型（Retrieve Model）：成本低，有一定推理能力，支持百万级别长上下文窗口的大语言模型
- 推理模型（Reasoning Model）：强推理，支持百万级别长上下文窗口的大语言模型
- 验证模型（Verify Model）：强推理
