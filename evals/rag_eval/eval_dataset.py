"""RAG evaluation dataset — questions, ground truth answers, and expected sources.

Based on the technical documentation in my-docs/sample.txt
and general knowledge questions for broader testing.
"""

EVAL_DATASET = [
    # --- 基于 sample.txt 的精确检索测试 ---
    {
        "id": 1,
        "question": "GPT Researcher 的 RAG 工作流程是怎样的？",
        "ground_truth": "RAG 工作流程包括：1.文档加载（PDF/DOCX/TXT通过DocumentLoader）2.文本分块（RecursiveCharacterTextSplitter，chunk_size=1000，overlap=200）3.Embedding生成（Memory类调用提供商）4.相似度过滤（EmbeddingsFilter，阈值0.35）5.上下文压缩（ContextualCompressionRetriever管道）",
        "expected_keywords": ["文档加载", "文本分块", "Embedding", "相似度过滤", "上下文压缩"],
    },
    {
        "id": 2,
        "question": "GPT Researcher 支持哪些搜索引擎？",
        "ground_truth": "支持 Tavily、DuckDuckGo、Google、Bing 等 16+ 检索器",
        "expected_keywords": ["Tavily", "DuckDuckGo", "Google", "Bing"],
    },
    {
        "id": 3,
        "question": "项目默认使用的快速 LLM 模型是什么？",
        "ground_truth": "FAST_LLM 默认使用 gpt-4o-mini",
        "expected_keywords": ["gpt-4o-mini", "FAST_LLM"],
    },
    {
        "id": 4,
        "question": "相似度阈值默认是多少？压缩阈值是多少？",
        "ground_truth": "SIMILARITY_THRESHOLD 默认 0.35，COMPRESSION_THRESHOLD 默认 8000 字符",
        "expected_keywords": ["0.35", "8000"],
    },
    {
        "id": 5,
        "question": "GPT Researcher 支持哪些报告类型？",
        "ground_truth": "支持 research_report（标准研究报告）、detailed_report（详细分析）、quick_report（快速摘要）、deep（深度研究模式）",
        "expected_keywords": ["research_report", "detailed_report", "quick_report", "deep"],
    },
    {
        "id": 6,
        "question": "GPT Researcher 的数据源模式有哪些？",
        "ground_truth": "web（网络搜索）、local（本地文档）、hybrid（网络+本地混合）、langchain_vectorstore（向量库检索）",
        "expected_keywords": ["web", "local", "hybrid", "vectorstore"],
    },
    {
        "id": 7,
        "question": "后端使用什么框架？前端有哪些选项？",
        "ground_truth": "后端使用 FastAPI + uvicorn，前端有静态前端和 Next.js 两种选项",
        "expected_keywords": ["FastAPI", "uvicorn", "Next.js"],
    },
    {
        "id": 8,
        "question": "文本分块的参数是什么？overlap 是多少？",
        "ground_truth": "使用 RecursiveCharacterTextSplitter，chunk_size=1000，overlap=200",
        "expected_keywords": ["1000", "200"],
    },
    # --- 跨段落/综合理解测试 ---
    {
        "id": 9,
        "question": "如果我要做复杂推理任务应该用哪个模型？做快速任务呢？",
        "ground_truth": "复杂推理用 STRATEGIC_LLM（默认 o4-mini），快速任务用 FAST_LLM（默认 gpt-4o-mini）",
        "expected_keywords": ["o4-mini", "STRATEGIC_LLM", "gpt-4o-mini", "FAST_LLM"],
    },
    {
        "id": 10,
        "question": "Embedding 模型支持哪些提供商？默认模型是什么？",
        "ground_truth": "支持 OpenAI、Cohere、Google、Ollama 等提供商，默认模型是 text-embedding-3-small",
        "expected_keywords": ["text-embedding-3-small", "OpenAI", "Cohere"],
    },
]
