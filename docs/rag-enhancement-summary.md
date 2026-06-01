# RAG Enhancement Module — 技术总结

> 基于 GPT Researcher (v0.14.7) 的 RAG 模块优化，涵盖全链路增强、持久化知识库、性能优化与 CRAG 基准评估。

---

## 一、项目背景

GPT Researcher 是一个自主深度研究 Agent，核心流程为：**搜索 → 爬取 → 上下文压缩 → 报告生成**。原有的上下文处理模块 (`ContextCompressor`) 存在以下瓶颈：

| 问题 | 原有实现 | 影响 |
|------|----------|------|
| 分块策略单一 | 固定 1000 字符滑动窗口 | 语义边界被切断 |
| 句子切分粗糙 | `(?<=[.!?])\s+` 简单正则 | U.S. / Dr. / $322.5 被误断 |
| 仅向量检索 | 单路 embedding 相似度过滤 | 关键词精确匹配能力弱 |
| 无查询优化 | 原始 query 直接检索 | 用户表述偏差导致漏检 |
| 无重排机制 | 检索结果直接输出 | 噪声 chunk 混入上下文 |
| 无去重逻辑 | 相似段落重复出现 | 浪费 token，降低报告质量 |
| 无持久化 | 每次研究重新处理所有文档 | 重复计算，无法积累知识 |
| 冗余 API 调用 | 每阶段独立 embed 同一文本 | 管线延迟 3-5x |

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
│        ↑              ↑            ↑           ↑             │
│        └──────────────┴────────────┴───────────┘             │
│              embedding 沿管线缓存复用（避免重复 API 调用）     │
│                                                              │
│  ┌──────────────┐              ┌─────────────────┐          │
│  │KnowledgeStore│              │  RAGAdapter     │          │
│  │ (Chroma 持久化)│              │ (兼容 ContextMgr)│          │
│  │  向量+BM25混合 │              └─────────────────┘          │
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

### 文件结构

```
rag_enhanced/
├── config.py                          # 统一配置 RAGConfig
├── pipeline.py                        # 全链路编排 RAGPipeline
├── adapter.py                         # 与 GPTResearcher 的兼容适配器
│
├── chunking/                          # 分块层
│   ├── base.py                        #   Chunk 数据类（含 embedding 缓存字段）+ BaseChunker ABC
│   ├── semantic.py                    #   SemanticChunker — 语义边界 + max_size + overlap
│   └── adaptive.py                    #   AdaptiveChunker — 多格式结构检测 + 自适应参数
│
├── retrieval/                         # 检索层
│   ├── base.py                        #   BaseRetriever ABC
│   ├── hybrid.py                      #   HybridRetriever — BM25 + 向量混合检索（embedding 缓存）
│   └── query_rewriter.py             #   QueryRewriter — 查询改写（Multi/HyDE/Auto）
│
├── reranking/                         # 重排层
│   ├── base.py                        #   BaseReranker ABC
│   ├── embedding_rerank.py            #   EmbeddingReranker — 复用 chunk embedding（零额外 API）
│   └── cross_encoder.py              #   CrossEncoderReranker — 交叉编码器重排
│
├── compression/                       # 压缩层
│   ├── base.py                        #   BaseCompressor ABC
│   └── context_aware.py              #   ContextAwareCompressor — 复用 embedding 去重 + Top-k
│
├── knowledge_store/                   # 持久化知识库
│   ├── base.py                        #   BaseKnowledgeStore ABC
│   └── chroma_store.py               #   ChromaKnowledgeStore — ChromaDB + 内存 BM25 + RRF
│
├── tools/                             # 工具接口
│   └── rag_query.py                   #   RAGQueryTool + 便捷函数
│
└── utils/
    └── text.py                        #   文本处理（缩写感知句子切分、余弦相似度）
```

---

## 三、各模块技术细节

### 3.1 Chunk 数据模型 — 五维评分 + Embedding 缓存

```python
@dataclass
class Chunk:
    content: str
    metadata: dict                          # source, title, chunk_index
    vector_score: float = 0.0               # 向量检索余弦相似度
    bm25_score: float = 0.0                 # BM25 关键词匹配分
    hybrid_score: float = 0.0               # RRF/加权融合分
    rerank_score: float = 0.0               # 重排精确分
    embedding: list[float] | None = None    # 缓存 embedding 向量（避免重复 API 调用）
```

**embedding 缓存机制**：检索阶段计算的 embedding 会写入 `embedding` 字段，沿管线传递给重排和压缩阶段。每个阶段先检查 `chunk.embedding is None`，已有则直接复用，跳过 API 调用。

**性能影响**：KB-only 路径从每题 4 次 embedding API 调用降至 1 次，延迟从 2.0s 降至 0.97s（2.1x 提速）。

### 3.2 缩写感知句子切分

**问题**：原实现 `re.split(r'(?<=[.!?])\s+', text)` 在 "U.S."、"Dr."、"$322.5"、"e.g." 处错误断句，导致 SemanticChunker 产生大量碎片 chunk。

**方案**：正边界检测 + 缩写白名单排除

```
1. 扫描所有 [.!?] + 空格 + 大写字母/引号 的位置（潜在句子边界）
2. 对每个边界，检查其前方文本是否匹配已知缩写模式：
   · 单字母缩写: U. S. (大写字母 + 句号)
   · 称谓缩写: Dr. Mr. Mrs. Prof. Rev. 等 40+ 个
   · 月份缩写: Jan. Feb. Mar. 等
   · 拉丁缩写: e.g. i.e. etc. vs. cf. al.
   · 省略号: ... (连续两个以上句号)
3. 排除缩写边界后，在剩余位置切分
```

**效果**：正确处理 "The U.S. economy grew 3.2% in Q1. Dr. Smith confirmed." → 2 句（而非 6+ 句碎片）。

### 3.3 语义分块（SemanticChunker）

**算法**：基于相邻句子 embedding 相似度的断点检测

```
文档 → 缩写感知拆句 → 批量 embed → 相邻余弦相似度 → 断点检测
    → 分组 → 合并碎片 → 句子级 overlap → 超限二次切分
```

三个新增特性：

| 特性 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| **max_chunk_size** | `max_chunk_size` | 2000 chars | 超限 chunk 用 `RecursiveCharacterTextSplitter` 二次切分，防止超大 chunk 超出 embedding 有效长度 |
| **sentence_overlap** | `sentence_overlap` | 1 句 | 相邻 chunk 共享末尾 N 个句子，overlap 不计入 max_chunk_size |
| **断点检测** | `breakpoint_threshold` | 0.3 | `similarity < mean - threshold × std` |

**断点判定示例**：10 个句子有 9 个相邻相似度。如果第 4-5 句之间相似度显著低于均值，则在此处切分，前 4 句为一个 chunk，后 6 句为另一个。

### 3.4 自适应分块（AdaptiveChunker）

**多格式结构化检测**（原仅支持 markdown `#`，现扩展为三种）：

| 检测方式 | 匹配条件 | 切分策略 |
|----------|----------|----------|
| **Markdown 标题** | ≥2 个 `#`~`######` | 按 `#{1,6}\s+` 正则切分 |
| **HTML 标题** | ≥2 个 `<h1>`~`<h6>` | BeautifulSoup 按 heading 标签切分，自动 strip HTML |
| **段落结构** | ≥3 个 `\n\n` 分隔 | 按空行切分，短段落合并 |

**工程优化**：

- **Splitter 缓存**：`_get_splitter(size, overlap)` 按参数缓存实例，避免每次切分重建
- **大 section 保护**：结构化切分后，超过 2000 字符的 section 自动用 `RecursiveCharacterTextSplitter` 二次切分
- **HTML 文档支持**：CRAG 等真实网页数据不再被当成无结构文本来切分

**参数自适应**：

| 文档类型 | 判定条件 | chunk_size | overlap |
|----------|----------|------------|---------|
| 结构化（任意格式） | 检测到标题/段落结构 | 按结构边界切分 | — |
| 短文档 | <2000 字符 | 500 | 50 |
| 长文档 | >5000 字符 | 1500 | 200 |
| 普通 | 其他 | 1000（默认） | 100 |

### 3.5 混合检索（HybridRetriever）

**双路检索 + embedding 缓存**：

```
Query ──┬── BM25 路──→ 关键词匹配排名
        │                    ↓
        └── 向量路──→ 语义相似度排名（embedding 写入 chunk.embedding）
                             ↓
                      融合（RRF 或加权）
                             ↓
                      hybrid_score 排序 → top_k
```

**embedding 缓存**：`_vector_search()` 计算的 embedding 直接存入 `chunk.embedding`，下游重排和压缩阶段直接复用。

**RRF（Reciprocal Rank Fusion）**：只看排名，不看分数绝对值

```
RRF_score(chunk) = Σ  1/(k + rank)    k=60（标准值）
```

### 3.6 查询改写（QueryRewriter）

| 策略 | 方式 | LLM 开销 | 场景 |
|------|------|----------|------|
| **Multi-Query**（默认） | LLM 生成 2-3 个替代查询 | 中 | 表述偏差修正 |
| **HyDE** | LLM 生成假设性答案用于 embedding | 高 | 语义匹配增强 |
| **Auto** | 先用原查询，不足时自动扩展 | 低→中 | 平衡效率 |
| None | 原查询直通 | 无 | 简单场景 |

### 3.7 重排（Reranker）— embedding 复用

**EmbeddingReranker**（默认，零额外 API 成本）：
```
query_embedding ← embed(query)
chunk_embedding ← chunk.embedding  ← 直接从缓存读取，不再调 API
rerank_score = cosine_similarity(query_embedding, chunk_embedding)
```

无缓存时退化为基础模式（调 API 获取），有缓存时跳过 `aembed_documents()` 调用。

### 3.8 上下文压缩（ContextAwareCompressor）— embedding 复用

```
输入 chunks → 快速路径判断（<2000字符且数量不超限？直接返回）
           → 检查 chunk.embedding → 有则复用，无则调 API
           → 成对余弦相似度去重（>0.85 的保留高分者）
           → 按 rerank_score > hybrid_score > vector_score 排序
           → 取 top_k → 格式化输出
```

### 3.9 持久化知识库（ChromaKnowledgeStore）

基于 ChromaDB 的向量持久化存储 + 内存 BM25 混合检索：

- **双路检索**：Chroma 向量搜索（持久化）+ 内存 BM25（`rank_bm25`，懒加载）→ RRF 融合
- **BM25 生命周期**：脏标记 + 懒重建。`add/delete` 标记脏，`retrieve` 触发重建。重建从 Chroma 加载全量文档，<100ms
- **Embedding 透出**：`retrieve()` 从 Chroma 取出已存储的 embedding，附到 Chunk 上供下游复用
- **Query embedding 控制**：自行调 `embeddings.aembed_query()` 再传 `query_embeddings` 给 Chroma，避免 Chroma 内部用默认 embedding 函数导致维度不匹配
- **去重入库**：`add_documents()` 自动清理同 source 的旧数据
- **便捷入库**：`ingest_local_docs(path)` 一行完成文件加载→分块→入库

### 3.10 Pipeline 双模式

| 模式 | 行为 | 场景 |
|------|------|------|
| `supplement` | 先查知识库，再查内存文档，合并去重 | GPTResearcher 研究 |
| `primary` | 仅查知识库 | Q&A 问答 |

### 3.11 RAGQueryTool — 轻量问答 + 工具接口

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

---

## 四、关键优化点总结

### 4.1 分块层优化

| 优化项 | 问题 | 方案 | 效果 |
|--------|------|------|------|
| 缩写感知句子切分 | `U.S.` `Dr.` `$322.5` 被误断句 | 正边界检测 + 缩写白名单排除 | SemanticChunker 不再产生碎片 chunk |
| max_chunk_size | 同主题长段落合成超大 chunk | 语义断点分组后二次切分 | 每个 chunk 控制在 2000 字符内 |
| sentence_overlap | 相邻 chunk 零共享，跨边界信息丢失 | 相邻 group 共享末尾 N 个句子 | 跨 chunk 边界查询不再丢失 |
| 多格式结构检测 | HTML 网页被当无结构文本 | 支持 markdown / HTML / 段落三种结构 | CRAG 网页数据正确按结构切分 |
| splitter 缓存 | 每次切分重建 TextSplitter 实例 | 按参数缓存 | 减少对象创建开销 |

### 4.2 全链路 Embedding 缓存

| 阶段 | 优化前 API 调用 | 优化后 API 调用 |
|------|----------------|----------------|
| HybridRetriever | embed(query) + embed(all chunks) | embed(query) + embed(missing chunks) |
| EmbeddingReranker | embed(query) + embed(all chunks) | embed(query) only（复用 chunk.embedding） |
| ContextAwareCompressor | embed(all chunks) | 0（全部复用） |
| **Enhanced 总计** | **5 次** | **2-3 次** |
| **KB-only 总计** | **4 次** | **1 次** |

**实测效果**（CRAG 50 题评估）：

| 系统 | 优化前延迟 | 优化后延迟 | 提速 |
|------|----------|----------|------|
| KB-only | 2.00s | **0.97s** | **2.1x** |
| Enhanced | 8.23s | **6.71s** | **1.2x** |

### 4.3 与原系统对比

| 维度 | 原有 ContextCompressor | rag_enhanced |
|------|----------------------|-------------|
| 句子切分 | 简单正则，缩写处误断 | 缩写感知正边界检测 |
| 分块 | 固定 1000 字符窗口 | 语义边界 + 自适应参数 + max_size + overlap |
| 结构检测 | 无 | markdown / HTML / 段落三级检测 |
| 检索 | 单路 embedding 过滤 | BM25 + 向量混合 + RRF 融合 |
| 查询优化 | 无 | Multi-Query / HyDE / Auto |
| 重排 | 无 | embedding 重排（零额外 API）/ cross-encoder |
| 去重 | 无 | 余弦相似度去重（>0.85） |
| 持久化 | 无（每次重新处理） | ChromaDB 向量库 + 内存 BM25 |
| Embedding 效率 | 每阶段独立计算 | 沿管线缓存复用，API 调用减少 60-75% |
| 工具化 | 仅内部调用 | 模块级函数供外部 Agent 调用 |

---

## 五、CRAG 基准评估

使用 Facebook Research 的 CRAG（Comprehensive RAG Benchmark）数据集评估，2706 个真实问题覆盖 5 个领域、8 种问题类型。

### 评估配置

- 数据集：CRAG Task 1 dev split（1371 题），采样 50 题（每领域 10 题）
- 系统：Original / Enhanced / KB-only
- 指标：Token Recall（gold answer token 在 context 中的覆盖率）、Phrase Recall、延迟

### 检索质量结果

| 系统 | Token Recall | Phrase Recall | 平均延迟 |
|------|-------------|---------------|---------|
| Original | 53.5% | 46.1% | 6.55s |
| Enhanced | 53.9% | 46.1% | 6.71s |
| **KB-only** | **54.7%** | 46.1% | **0.97s** |

### 按问题类型分析

| 问题类型 | Original | Enhanced | KB-only | 说明 |
|---------|----------|----------|---------|------|
| **set** | 75% | **86%** | 79% | Enhanced 混合检索优势最大 |
| **aggregation** | 79% | **82%** | **82%** | 跨段落综合题，多路检索有帮助 |
| **comparison** | 65% | 67% | **68%** | 略有改善 |
| multi-hop | 91% | 91% | 91% | 都很好 |
| false_premise | 8% | 0% | 8% | 全部低（问题前提错误，检索无法解决） |
| simple | 44% | 44% | 44% | 无差异 |

**关键发现**：Enhanced 在 `set`（+11%）和 `aggregation`（+3%）类型上优势明显——这类问题需要在多文档中找多个相关片段，BM25+向量混合检索正好发挥作用。

---

## 六、测试覆盖

83 个测试全部通过，覆盖所有模块：

```
tests/test_rag_enhanced/
├── test_chunking.py          # 句子切分 + SemanticChunker + AdaptiveChunker (29 tests)
├── test_retrieval.py         # HybridRetriever RRF/加权融合 (4 tests)
├── test_query_rewriter.py    # QueryRewriter Multi/HyDE/Auto (4 tests)
├── test_reranking.py         # EmbeddingReranker + CrossEncoderReranker (6 tests)
├── test_compression.py       # ContextAwareCompressor 去重/快速路径 (5 tests)
├── test_pipeline.py          # RAGPipeline 端到端 (4 tests)
├── test_adapter.py           # RAGAdapter 兼容性 (4 tests)
├── test_knowledge_store.py   # ChromaKnowledgeStore 全生命周期 + BM25 (11 tests)
├── test_rag_query.py         # RAGQueryTool + 便捷函数 (6 tests)
└── test_integration.py       # KnowledgeStore + Pipeline 集成 (6 tests)
```

---

## 七、使用方式

### 7.1 接入 GPTResearcher

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

### 7.2 独立使用 — 知识库问答

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

### 7.3 外部 Agent 调用

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

## 八、技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.11+ (全异步 asyncio) |
| 向量数据库 | ChromaDB（嵌入式，自动持久化，cosine distance） |
| Embedding | LangChain embeddings 接口（支持 OpenAI/Ollama 等 20+ 提供商） |
| BM25 | rank_bm25 库（内存，懒加载，脏标记重建） |
| 文本分块 | langchain_text_splitters + 自定义语义/自适应分块 |
| Cross-Encoder | sentence-transformers（可选） |
| HTML 解析 | BeautifulSoup4（结构化检测 + HTML 文档切分） |
| 评估数据集 | Facebook Research CRAG（2706 题，5 领域，8 问题类型） |
| 测试 | pytest + pytest-asyncio (strict mode) |

---

## 九、Git 提交历史

```
435a8624 docs: add RAG enhancement technical summary
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
