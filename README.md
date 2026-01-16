# RAG System for Existing Knowledge Base

A production-ready Retrieval-Augmented Generation (RAG) system that provides intelligent question-answering capabilities over a company knowledge base using advanced chunking strategies and comprehensive evaluation metrics.

## 🌟 Features

- **Intelligent Document Processing**: Automatically splits documents into semantically meaningful chunks with headlines and summaries using LLM-powered preprocessing
- **Advanced Retrieval**: Two-stage retrieval pipeline with semantic search and LLM-based reranking
- **Comprehensive Evaluation**: Built-in evaluation framework with MRR, nDCG, and LLM-as-a-judge metrics
- **Multi-domain Knowledge Base**: Handles company info, employee data, product details, and contracts
- **Parallel Processing**: Multi-threaded document ingestion for faster preprocessing
- **Robust Error Handling**: Automatic retries with exponential backoff for API calls

## 📁 Project Structure

```
RAG_github/
├── ingest.py                   # Document preprocessing and vectorization
├── evaluation_func/
│   ├── answer.py              # RAG query pipeline with reranking
│   ├── eval.py                # Evaluation metrics (MRR, nDCG, LLM judge)
│   ├── test.py                # Test case utilities
│   └── tests.jsonl            # Evaluation test dataset
├── evaluation.ipynb           # Jupyter notebook for running evaluations
├── knowledge-base/            # Source documents
│   ├── company/              # Company information
│   ├── contracts/            # Contract documents
│   ├── employees/            # Employee profiles
│   └── products/             # Product documentation
├── preprocessed_db/          # ChromaDB vector database (generated)
└── pyproject.toml            # Project dependencies

```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RAG_github.git
cd RAG_github
```

2. Install dependencies:
```bash
uv sync
```

3. Set up environment variables:
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Data Ingestion

Process documents and create the vector database:

```bash
uv run ingest.py
```

This will:
1. Load all markdown documents from `knowledge-base/`
2. Split documents into intelligent chunks using GPT-4.1-nano
3. Generate headlines and summaries for each chunk
4. Create embeddings using `text-embedding-3-large`
5. Store everything in ChromaDB

**Note**: Adjust `WORKERS` in `ingest.py` if you encounter rate limits (default: 3).

## 💬 Usage

### Command-Line Interface

Run a single evaluation test:

```bash
uv run evaluation_func/eval.py <test_number>
```

Example:
```bash
uv run evaluation_func/eval.py 0
```

This will display:
- Question and reference answer
- Retrieval metrics (MRR, nDCG, keyword coverage)
- Generated answer with LLM judge scores

### Programmatic Usage

```python
from evaluation_func.answer import answer_question

# Ask a question
answer, context_docs = answer_question("What products does Insurellm offer?")
print(answer)
```

### Jupyter Notebook

For comprehensive evaluation, use the provided notebook:

```bash
jupyter notebook evaluation.ipynb
```

The notebook provides:
- Batch retrieval evaluation across all test cases
- LLM-as-a-judge answer quality assessment
- Category-wise performance analysis
- Summary statistics and visualizations

## 📊 Evaluation Metrics

### Retrieval Metrics

- **MRR (Mean Reciprocal Rank)**: Measures how quickly relevant documents are retrieved
- **nDCG (Normalized Discounted Cumulative Gain)**: Evaluates ranking quality
- **Keyword Coverage**: Percentage of expected keywords found in retrieved context

### Answer Quality Metrics (LLM-as-a-Judge)

- **Accuracy** (1-5): Factual correctness compared to reference answer
- **Completeness** (1-5): Coverage of all aspects in the reference answer
- **Relevance** (1-5): Direct addressing of the question without extra information

## 🔧 Configuration

### Key Parameters in `ingest.py`

```python
MODEL = "openai/gpt-4.1-nano"          # LLM for chunk generation
embedding_model = "text-embedding-3-large"  # Embedding model
AVERAGE_CHUNK_SIZE = 100               # Target words per chunk
WORKERS = 3                            # Parallel workers
```

### Key Parameters in `answer.py`

```python
MODEL = "openai/gpt-4.1-nano"          # LLM for reranking & answering
RETRIEVAL_K = 20                       # Initial retrieval count
FINAL_K = 10                           # Final reranked chunks
```

## 🏗️ Architecture

### 1. Document Ingestion Pipeline

```
Documents → LLM Chunking → Headline/Summary Generation → Embeddings → ChromaDB
```

Each chunk contains:
- **Headline**: Brief query-optimized heading
- **Summary**: Condensed content for quick understanding
- **Original Text**: Complete original content

### 2. Query Pipeline

```
Query → Embedding → Retrieve Top-K → LLM Reranking → Generate Answer
```

Two-stage retrieval ensures both recall and precision.

## 🧪 Testing

The project includes a comprehensive test suite in `tests.jsonl` covering:
- Direct factual questions
- Multi-document queries
- Category-specific questions (products, employees, contracts, etc.)

Run all evaluations:

```python
# In evaluation.ipynb
for test, result, progress in evaluate_all_retrieval():
    print(f"Test {progress*100:.0f}% complete")
```

## 📝 Adding New Documents

1. Add markdown files to appropriate `knowledge-base/` subfolder
2. Re-run ingestion:
```bash
uv run ingest.py
```

The system will automatically process new documents.

