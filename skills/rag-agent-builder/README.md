# RAG Agent Builder - Code Structure

This skill uses supporting Python files to keep documentation lean and maintainable.

## Directory Structure

```
rag-agent-builder/
├── SKILL.md                      # Main documentation (concepts, patterns)
├── README.md                     # This file
├── examples/                     # Implementation examples
│   ├── basic_rag.py              # Simple RAG pipeline
│   ├── retrieval_strategies.py   # Hybrid search, reranking, filtering
│   └── agentic_rag.py            # Agent-controlled retrieval
└── scripts/                      # Utility modules
    ├── embedding_management.py   # Embedding generation and caching
    ├── vector_db_manager.py      # Vector database abstraction
    └── rag_evaluation.py         # Retrieval and answer quality metrics
```

## Running Examples

### 1. Basic RAG
```bash
python examples/basic_rag.py
```
Simplest RAG implementation - chunk documents, embed, retrieve, generate.

### 2. Advanced Retrieval Strategies
```bash
python examples/retrieval_strategies.py
```
Hybrid search combining keyword and semantic search with reranking.

### 3. Agentic RAG
```bash
python examples/agentic_rag.py
```
Agent-controlled retrieval with iterative refinement for complex questions.

## Using the Utilities

### Embedding Management
```python
from scripts.embedding_management import EmbeddingManager, EmbeddingQualityAssessment

manager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
embeddings = manager.generate_embeddings(texts)
normalized = manager.normalize_embeddings(embeddings)

quality = EmbeddingQualityAssessment.embedding_distribution_quality(embeddings)
print(f"Embedding quality: {quality['quality']}")
```

### Vector Database Management
```python
from scripts.vector_db_manager import VectorDBFactory, VectorDBManager

# Create in-memory database
db = VectorDBFactory.create_db("in_memory")
manager = VectorDBManager(db)

# Add documents
doc_ids = manager.add_documents(texts, embeddings)

# Search
results = manager.search(query_embedding, k=5)

# Database info
info = manager.get_database_info()
```

### RAG Evaluation
```python
from scripts.rag_evaluation import RAGEvaluator, RetrievalMetrics

evaluator = RAGEvaluator()

# Evaluate retrieval
retrieval_eval = evaluator.evaluate_retrieval(retrieved_docs, relevant_docs, query)

# Evaluate answer
answer_eval = evaluator.evaluate_answer(answer, context, query)

# Complete pipeline evaluation
pipeline_eval = evaluator.evaluate_rag_pipeline(
    query, retrieved, relevant, answer, context
)

# Get summary
summary = evaluator.get_evaluation_summary()
```

## Integration with SKILL.md

- SKILL.md contains conceptual information, patterns, and best practices
- Code examples are in `examples/` for clarity and reusability
- Utilities are in `scripts/` for modular components
- This keeps token costs low while maintaining full functionality

## Architecture Patterns Covered

1. **Basic RAG** - Simple chunk-embed-retrieve-generate pipeline
2. **Agentic RAG** - Agent makes intelligent retrieval decisions
3. **Hybrid Search** - Combines keyword and semantic search
4. **Retrieval Refinement** - Reranking and filtering
5. **Context Management** - Fit documents within token limits
6. **Quality Evaluation** - Metrics for retrieval and answer quality

## Models and Technologies Supported

- **Embedding Models**: All HuggingFace sentence-transformers compatible
- **Vector DBs**: In-memory (reference), Chroma, Pinecone, Weaviate, Qdrant
- **Frameworks**: LangChain, LlamaIndex, HayStack compatible
- **LLMs**: Any API-compatible LLM (OpenAI, local, etc.)

## Key Features

- **Token Efficient**: Modular code structure reduces LLM context usage
- **Production Ready**: Includes evaluation metrics and quality assessment
- **Framework Agnostic**: Works with any embedding or vector DB
- **Iterative Improvement**: Agent-based RAG for complex queries
- **Quality Focused**: Built-in metrics for retrieval and answer quality

## Next Steps

1. Choose embedding model for your domain
2. Prepare and chunk your documents
3. Select vector database for storage
4. Implement retrieval strategy (basic, hybrid, or agentic)
5. Evaluate with provided metrics
6. Iterate based on quality scores
