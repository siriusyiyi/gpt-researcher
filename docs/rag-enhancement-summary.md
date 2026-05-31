# RAG Enhancement Module — 技术总结

> 基于 GPT Researcher (v0.14.7) 的 RAG 模块优化，涵盖全链路增强与持久化知识库构建。

---

## 一、项目背景

GPT Researcher 是一个自主深度研究 Agent，核心流程为：**搜索 → 爬取 → 上下文压缩 → 报告生成**。原有的上下文处理模块 (`ContextCompressor`) 存在以下瓶颈：

| 问题 | 原有实现 | 影响 |
|------|----------|------|
| 分块策略单一 | 固定 1000 字符滑动窗口 | 语义边界被切断 |
| 仅向量检索 | 单路 embedding 相似度过滤 | 关键词精确匹配能力弱 |
| 无查询优化 | 原始 query 直接检索 | 用户表述偏差导致漏检 |
| 无重排机制 | 检索结果直接输出 | 噪声 chunk 混入上下文 |
| 无去重逻辑 | 相似段落重复出现 | 浪费 token，降低报告质量 |
| 无持久化 | 每次研究重新处理所有文档 | 重复计算，无法积累知识 |

本模块 (`rag_enhanced/`) 作为独立目录实现，通过 `RAGAdapter` 与原有系统无缝对接，保持向后兼容。

---

## 二、系统架构

### 整体流程

```
┌──────────────────────────────────────────────────────────────┐
│                        rag_enhanced/                         │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Chunking │→ │ Retrieval│→ │ Reranking│→ │ Compression│  │
│  │          │  │          │  │          │  │            │  │
│  │·Semantic │  │·Hybrid   │  │·Embedding│  │·Dedup      │  │
│  │·Adaptive │  │  BM25+Vec│  │·CrossEnc │  │·Top-k      │  │
│  └──────────┘  └──────────┘  └──────────┘  └────────────┘  │
│        ↑                                ↓                   │
│  ┌──────────────┐              ┌─────────────────┐          │
│  │KnowledgeStore│              │  RAGAdapter     │          │
│  │ (Chroma 持久化)│              │ (兼容 ContextMgr)│          │
│  │              │              └─────────────────┘          │
│  └──────────────┘                                           │
│        ↑                                                    │
│  ┌──────────────────────────────────────┐                   │
│  │         RAGQueryTool                 │                   │
│  │  query → rewrite → retrieve → rerank │                   │
│  │  → compress → LLM answer + citations │                   │
│  │                                      │                   │
│  │  对外工具函数（供外部 Agent 调用）:     │                   │
│  │  · rag_query(question) → dict        │                   │
│  │  · knowledge_add(documents) → int    │                   │
│  │  · knowledge_search(query) → list    │                   │
│  └──────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────────┘
```

### 文件结构（26 个源文件，1730 行；12 个测试文件，1266 行）

```
rag_enhanced/
├── config.py                          # 统一配置 RAGConfig
├── pipeline.py                        # 全链路编排 RAGPipeline
├── adapter.py                         # 与 GPTResearcher 的兼容适配器
│
├── chunking/                          # 分块层
│   ├── base.py                        #   Chunk 数据类 + BaseChunker ABC
│   ├── semantic.py                    #   SemanticChunker — 语义边界检测
│   └── adaptive.py                    #   AdaptiveChunker — 按文档特征自适应
│
├── retrieval/                         # 检索层
│   ├── base.py                        #   BaseRetriever ABC
│   ├── hybrid.py                      #   HybridRetriever — BM25 + 向量混合检索
│   └── query_rewriter.py             #   QueryRewriter — 查询改写（Multi/HyDE/Auto）
│
├── reranking/                         # 重排层
│   ├── base.py                        #   BaseReranker ABC
│   ├── embedding_rerank.py            #   EmbeddingReranker — 余弦相似度重排
│   └── cross_encoder.py              #   CrossEncoderReranker — 交叉编码器重排
│
├── compression/                       # 压缩层
│   ├── base.py                        #   BaseCompressor ABC
│   └── context_aware.py              #   ContextAwareCompressor — 去重 + Top-k
│
├── knowledge_store/                   # 持久化知识库
│   ├── base.py                        #   BaseKnowledgeStore ABC
│   └── chroma_store.py               #   ChromaKnowledgeStore — ChromaDB 实现
│
├── tools/                             # 工具接口
│   └── rag_query.py                   #   RAGQueryTool + 便捷函数
│
└── utils/
    └── text.py                        #   文本处理工具函数
```

---

## 三、各模块技术细节

### 3.1 Chunk 数据模型 — 四维评分体系

```python
@dataclass
class Chunk:
    content: str
    metadata: dict                          # source, title, chunk_index
    vector_score: float = 0.0               # 向量检索余弦相似度
    bm25_score: float = 0.0                 # BM25 关键词匹配分
    hybrid_score: float = 0.0               # RRF/加权融合分
    rerank_score: float = 0.0               # 重排精确分
```

四个分数由不同阶段写入，互不覆盖，下游可综合决策。这是区别于传统单分数 RAG 系统的设计。

### 3.2 语义分块（SemanticChunker）

**算法**：基于相邻句子 embedding 相似度的断点检测

```
文档 → 拆句 → 批量 embed → 计算相邻余弦相似度 → 检测断点 → 分组 → 合并碎片
```

断点判定：`similarity[i] < mean - threshold × std`（低于均值减阈值倍标准差）

**解决的问题**：固定窗口分块会在句子中间切断，破坏语义完整性。语义分块确保每个 chunk 内的话题连贯。

### 3.3 自适应分块（AdaptiveChunker）

根据文档特征自动调整分块参数：

| 文档类型 | 判定条件 | chunk_size | overlap |
|----------|----------|------------|---------|
| 结构化（有标题） | ≥2 个标题 | 按标题边界切分 | — |
| 短文档 | <2000 字符 | 500 | 50 |
| 长文档 | >5000 字符 | 1500 | 200 |
| 普通 | 其他 | 1000（默认） | 100 |

### 3.4 混合检索（HybridRetriever）

**双路检索**：

```
Query ──┬── BM25 路──→ 关键词匹配排名
        │                    ↓
        └── 向量路──→ 语义相似度排名
                             ↓
                      融合（RRF 或加权）
                             ↓
                      hybrid_score 排序 → top_k
```

**RRF（Reciprocal Rank Fusion）**：只看排名，不看分数绝对值

```
RRF_score(chunk) = Σ  1/(k + rank)    k=60（标准值）
```

**核心优势**：BM25 分数（词频统计值）和向量分数（余弦相似度）尺度不同，无法直接比较。RRF 完全回避了分数对齐问题，天然兼容异构检索。

### 3.5 查询改写（QueryRewriter）

| 策略 | 方式 | LLM 开销 | 场景 |
|------|------|----------|------|
| **Multi-Query**（默认） | LLM 生成 2-3 个替代查询 | 中 | 表述偏差修正 |
| **HyDE** | LLM 生成假设性答案用于 embedding | 高 | 语义匹配增强 |
| **Auto** | 先用原查询，不足时自动扩展 | 低→中 | 平衡效率 |
| None | 原查询直通 | 无 | 简单场景 |

Auto 模式"不足"判定：chunk 数量 < `min_retrieval_results` 或最高分 < `min_top_score`。

### 3.6 重排（Reranker）

**EmbeddingReranker**（默认，零额外成本）：
```
query_embedding ← embed(query)
chunk_embedding ← embed(chunk)       # 复用已有 embedding
rerank_score = cosine_similarity(query_embedding, chunk_embedding)
```

**CrossEncoderReranker**（更精确，需额外模型）：
```
rerank_score = CrossEncoder.predict([query, chunk])
# query 和 chunk 一起进入 Transformer 注意力层
# 能捕捉词级别的交互关系
```

**级联设计**：检索阶段用 RRF 粗筛（快），重排阶段用精确模型精选（准）。

### 3.7 上下文压缩（ContextAwareCompressor）

```
输入 chunks → 快速路径判断（<2000字符且数量不超限？直接返回）
           → embedding 所有 chunk
           → 成对余弦相似度去重（>0.85 的保留高分者）
           → 按 rerank_score > hybrid_score > vector_score 排序
           → 取 top_k → 格式化输出
```

输出格式：`Source: url\nTitle: title\nContent: text`

### 3.8 持久化知识库（ChromaKnowledgeStore）

基于 ChromaDB 的向量持久化存储：

- **存储**：Chunk 内容 + 元数据（source/title/doc_type）+ 自动 embedding
- **检索**：余弦距离相似度搜索（`hnsw:space=cosine`）
- **去重入库**：`add_documents()` 自动清理同 source 的旧数据
- **生命周期**：`add → retrieve → delete → list_sources`
- **便捷入库**：`ingest_local_docs(path)` 一行完成文件加载→分块→入库
- **持久化**：数据写入磁盘，跨会话可用

### 3.9 Pipeline 双模式

| 模式 | 行为 | 场景 |
|------|------|------|
| `supplement` | 先查知识库，再查内存文档，合并去重 | GPTResearcher 研究 |
| `primary` | 仅查知识库 | Q&A 问答 |

### 3.10 RAGQueryTool — 轻量问答 + 工具接口

**完整问答流程**：
```
question → QueryRewriter(可选) → KnowledgeStore.retrieve(并行) → 去重
         → EmbeddingReranker → ContextAwareCompressor → LLM 生成带引用回答
```

**对外工具函数**（供外部 Agent 调用）：
```python
from rag_enhanced.tools import rag_query, knowledge_add, knowledge_search

result = await rag_query("问题")           # → {"answer": ..., "sources": [...]}
count = await knowledge_add([docs])        # → 入库 chunk 数
chunks = await knowledge_search("关键词")   # → [{content, source, score}]
```

### 3.11 RAGAdapter — 兼容层

```python
# 一行替换原有 ContextManager
researcher.context_manager = RAGAdapter(researcher)
```

保持 `get_similar_content_by_query()` 签名不变，内部走增强 pipeline。

---

## 四、关键优化点总结

### 4.1 全链路优化

| 优化项 | 技术手段 | 效果 |
|--------|----------|------|
| 语义感知分块 | embedding 相似度断点检测 | chunk 语义完整性 ↑ |
| 自适应分块 | 按文档长度/结构动态调参 | 长文档不碎片，短文档不浪费 |
| 混合检索 | BM25 + 向量 + RRF 融合 | 关键词精确匹配 + 语义泛化 |
| 查询改写 | Multi-Query/HyDE/Auto | 用户表述偏差修正 |
| 精确重排 | embedding 重排（零成本）/ cross-encoder | 噪声过滤 ↑ |
| 上下文压缩 | 跨 chunk 去重 + top-k 选择 | token 效率 ↑ |
| 持久化知识库 | ChromaDB + cosine distance | 避免重复处理，知识积累 |

### 4.2 工程优化

| 优化项 | 实现 |
|--------|------|
| 入库去重 | `add_documents()` 自动清理同 source 旧数据 |
| 分数精确 | Chroma 使用 cosine distance，`1-score` 即余弦相似度 |
| 并行检索 | 多查询变体 `asyncio.gather` 并行，单查询无额外开销 |
| 向后兼容 | RAGAdapter 保持 ContextManager 接口不变 |
| 独立部署 | 模块独立于 GPTResearcher，可单独作为工具使用 |

### 4.3 与原系统对比

| 维度 | 原有 ContextCompressor | rag_enhanced |
|------|----------------------|-------------|
| 分块 | 固定 1000 字符窗口 | 语义边界检测 + 自适应参数 |
| 检索 | 单路 embedding 过滤 | BM25 + 向量混合 + RRF 融合 |
| 查询优化 | 无 | Multi-Query / HyDE / Auto |
| 重排 | 无 | embedding 重排 / cross-encoder |
| 去重 | 无 | 余弦相似度去重（>0.85） |
| 持久化 | 无（每次重新处理） | ChromaDB 向量库 |
| 工具化 | 仅内部调用 | 模块级函数供外部 Agent 调用 |
| LLM 依赖 | 每次压缩都调 embedding | 仅查询改写用 LLM（可选） |

---

## 五、测试覆盖

58 个测试全部通过，覆盖所有模块：

```
tests/test_rag_enhanced/
├── test_chunking.py          # Chunk 数据模型 + SemanticChunker (6 tests)
├── test_adaptive.py          # AdaptiveChunker 自适应逻辑 (4 tests)
├── test_retrieval.py         # HybridRetriever RRF/加权融合 (4 tests)
├── test_query_rewriter.py    # QueryRewriter Multi/HyDE/Auto (4 tests)
├── test_reranking.py         # EmbeddingReranker + CrossEncoderReranker (6 tests)
├── test_compression.py       # ContextAwareCompressor 去重/快速路径 (5 tests)
├── test_pipeline.py          # RAGPipeline 端到端 (4 tests)
├── test_adapter.py           # RAGAdapter 兼容性 (4 tests)
├── test_knowledge_store.py   # ChromaKnowledgeStore 全生命周期 (9 tests)
├── test_rag_query.py         # RAGQueryTool + 便捷函数 (6 tests)
└── test_integration.py       # KnowledgeStore + Pipeline 集成 (6 tests)
```

---

## 六、使用方式

### 6.1 接入 GPTResearcher

```python
from rag_enhanced.adapter import RAGAdapter
from rag_enhanced.config import RAGConfig

# 基础接入（替换 ContextManager）
researcher.context_manager = RAGAdapter(researcher)

# 启用知识库（本地文档持久化）
config = RAGConfig(enable_knowledge_store=True)
researcher.context_manager = RAGAdapter(researcher, config=config)
await researcher.context_manager.ingest_local_docs("./my-docs")
```

### 6.2 独立使用 — 知识库问答

```python
from rag_enhanced.tools import rag_query, knowledge_add, knowledge_search

# 添加文档到知识库
await knowledge_add([
    {"raw_content": "文档内容...", "url": "source.txt", "title": "标题"}
])

# 知识库问答（带 LLM 回答）
result = await rag_query("问题", store_path="./knowledge_store")
print(result["answer"])    # LLM 生成的带引用回答
print(result["sources"])   # 引用来源列表

# 纯检索（不调 LLM，最快）
chunks = await knowledge_search("关键词", top_k=5, store_path="./knowledge_store")
```

### 6.3 外部 Agent 调用

```python
# 在你的其他项目中直接导入使用
from rag_enhanced.tools import rag_query, knowledge_search, knowledge_add

# Agent 工具注册示例
tools = [
    {"name": "knowledge_search", "func": knowledge_search, "description": "搜索知识库"},
    {"name": "knowledge_add", "func": knowledge_add, "description": "添加文档到知识库"},
    {"name": "rag_query", "func": rag_query, "description": "知识库问答"},
]
```

---

## 七、技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.11+ (全异步 asyncio) |
| 向量数据库 | ChromaDB（嵌入式，自动持久化） |
| Embedding | LangChain embeddings 接口（支持 OpenAI/Ollama 等 20+ 提供商） |
| BM25 | rank_bm25 库 |
| 文本分块 | langchain_text_splitters |
| Cross-Encoder | sentence-transformers（可选） |
| 测试 | pytest + pytest-asyncio (strict mode) |

---

## 八、Git 提交历史

```
5869bf36 fix: ingestion dedup, cosine distance, parallel retrieval
23787c0a feat: add Chroma-based knowledge store and RAG query tools
a65d3cc3 fix(rag_enhanced): relax test assertions for negative cosine similarity
c4835230 feat(rag_enhanced): implement RAGPipeline orchestration and RAGAdapter
8c87a77f feat(rag_enhanced): implement ContextAwareCompressor with dedup and fast path
be73a605 feat(rag_enhanced): implement EmbeddingReranker and CrossEncoderReranker
f3e90678 feat(rag_enhanced): implement QueryRewriter with multi-query, HyDE, auto modes
4c2ec14f feat(rag_enhanced): implement HybridRetriever with BM25+vector RRF fusion
883cb876 feat(rag_enhanced): implement AdaptiveChunker with length/structure detection
8fdabd38 feat(rag_enhanced): implement SemanticChunker with embedding breakpoints
c14a314b feat(rag_enhanced): add foundation — Chunk dataclass, RAGConfig, text utils
```
