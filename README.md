# InsightForge: AI-Powered Business Intelligence Assistant

## Overview

InsightForge is an AI-powered Business Intelligence Assistant that transforms business data into actionable insights through natural language interaction. Built with OpenAI's GPT-4, LangGraph, and Chroma vector database, it enables non-technical users to analyze data through simple conversations.

## Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **Intelligent RAG System**: Retrieval-Augmented Generation for accurate, data-grounded responses
- **Real-time Visualizations**: Interactive charts for sales trends, product performance, and demographics
- **Memory-Enabled Conversations**: Context-aware chat that remembers previous interactions
- **Advanced Statistics**: Automatic extraction of sales metrics, product analysis, and customer insights
- **Data Explorer**: Browse and filter your data directly in the interface
- **Model Evaluation**: Built-in quality assessment and testing

## Quick Start

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- pip package manager

### Installation

1. **Clone or navigate to the project**:
```bash
cd /path/to/capstone
```

2. **Create and activate virtual environment**:
```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**:

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your-api-key-here
MODEL_NAME=gpt-4
```

5. **Run the application**:
```bash
streamlit run app.py
```

6. **Open your browser** to `http://localhost:8501`

## Usage

### Uploading Data

1. Click the file uploader in the sidebar
2. Select your CSV file (e.g., `data/raw/sales_data.csv`)
3. Wait for the knowledge base creation (~30 seconds)
4. Start chatting!

### Expected Data Format

Your CSV should include columns such as:
- **Date**: Transaction date (e.g., 2024-01-01)
- **Product**: Product name
- **Sales/Revenue**: Sales amount
- **Region**: Geographic region
- **Customer demographics**: Age, Gender, etc.

Example:
```csv
Date,Product,Region,Sales,Customer_Age,Customer_Gender
2024-01-01,Widget A,North,1500,35,M
2024-01-02,Widget B,South,2300,42,F
```

### Example Queries

**Sales Analysis:**
- "What is the total sales?"
- "Show me the sales trend"
- "Which month had the highest sales?"

**Product Insights:**
- "Which product performs best?"
- "Compare the top 3 products"
- "What are the product sales by region?"

**Customer Analysis:**
- "What is the average customer age?"
- "Show me customer demographics"
- "Which age group spends the most?"

**Regional Performance:**
- "Which region has the best performance?"
- "Compare sales across regions"

## Architecture

### System Components

```
InsightForge/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
│
├── src/
│   ├── data/
│   │   ├── loader.py          # CSV loading with PyArrow normalization
│   │   └── knowledge_base.py  # Chroma vector database management
│   │
│   ├── models/
│   │   ├── llm_config.py      # OpenAI LLM configuration
│   │   ├── rag_system.py      # LangGraph RAG pipeline
│   │   ├── custom_retriever.py # Statistical data extraction
│   │   └── memory_manager.py  # Conversation memory
│   │
│   ├── utils/
│   │   ├── prompts.py         # Prompt engineering templates
│   │   └── evaluation.py      # Model evaluation metrics
│   │
│   └── visualization/
│       └── charts.py          # Plotly visualization components
│
├── data/
│   └── raw/                   # Your CSV data files
│
├── chroma_db/                 # Vector database (auto-created)
│
└── .streamlit/
    └── config.toml            # Streamlit configuration
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | OpenAI GPT-4 | Natural language understanding and generation |
| **RAG Framework** | LangGraph | Stateful conversation workflows |
| **Vector Database** | Chroma | Document storage and similarity search |
| **Embeddings** | OpenAI text-embedding-3-small | Vector representations |
| **Data Processing** | Pandas + NumPy | Data manipulation and analysis |
| **Visualization** | Plotly | Interactive charts |
| **UI Framework** | Streamlit | Web application interface |
| **Memory** | Custom deque-based | Conversation context tracking |

### Data Flow

1. **Upload** → CSV file uploaded via Streamlit
2. **Normalize** → DataLoader converts all columns to consistent types (strings/numbers)
3. **Vectorize** → KnowledgeBase creates document embeddings with Chroma
4. **Query** → User asks question in natural language
5. **Retrieve** → RAG system fetches relevant documents + statistics
6. **Generate** → LLM produces answer based on retrieved context
7. **Display** → Response shown in chat interface with visualizations

## Core Features Explained

### 1. Smart Data Loading

The DataLoader automatically:
- Reads CSV with all columns as strings first
- Detects column types (dates, numbers, text)
- Normalizes dates to ISO format (YYYY-MM-DD)
- Ensures PyArrow compatibility for Streamlit

**Key Innovation**: Prevents mixed-type columns that cause display errors.

### 2. Vector Knowledge Base

Uses Chroma vector database to:
- Create document chunks from your data
- Generate embeddings for semantic search
- Store metadata for filtering
- Enable fast similarity queries

**Documents Created**:
- Dataset summary
- Time period aggregations (monthly)
- Product-level statistics
- Regional breakdowns

### 3. Custom Business Retriever

Combines two retrieval strategies:
1. **Vector Search**: Semantic similarity using Chroma
2. **Statistical Extraction**: Direct data queries based on intent

**Intent Classification**:
- Sales queries → Extract total, average, trends
- Product queries → Top sellers, comparisons
- Regional queries → Geographic performance
- Customer queries → Demographics, segmentation
- Time queries → Temporal trends, seasonality

### 4. LangGraph RAG Pipeline

Multi-node graph workflow:
```
Query → Retrieve → Generate → Format → Response
```

**Nodes**:
- `retrieve`: Fetch relevant documents and statistics
- `generate`: LLM creates response with context
- `format`: Structure output for display

**Benefits**:
- Stateful conversations
- Context preservation
- Error recovery
- Streaming support

### 5. Conversation Memory

Sliding window memory (last 5 exchanges):
- Tracks user questions
- Remembers AI responses
- Maintains conversation context
- Enables follow-up questions

### 6. Interactive Visualizations

Built-in chart types:
- **Sales Trend**: Time-series line charts
- **Product Comparison**: Horizontal bar charts
- **Regional Analysis**: Dual bar charts (total vs. average)
- **Customer Demographics**: Age histogram + gender pie chart
- **Summary Dashboard**: Multi-panel overview

All charts are:
- Interactive (zoom, pan, hover)
- Exportable as PNG/HTML
- Responsive to screen size

## Configuration

### Environment Variables

Edit `.env` file:

```bash
# Required
OPENAI_API_KEY=sk-your-api-key-here

# Optional
MODEL_NAME=gpt-4              # Or gpt-3.5-turbo for faster responses
TEMPERATURE=0.0               # 0.0 for factual, 0.7 for creative
```

### Streamlit Settings

Edit `.streamlit/config.toml` for:
- Theme customization
- Server port (default: 8501)
- File upload limits
- Caching behavior

## API Reference

### Key Classes

#### `DataLoader`
```python
from src.data.loader import DataLoader

loader = DataLoader()
df = loader.load_data("path/to/data.csv")
metadata = loader.get_metadata()
```

#### `KnowledgeBase`
```python
from src.data.knowledge_base import KnowledgeBase

kb = KnowledgeBase(persist_directory="./chroma_db")
vector_store = kb.create_from_dataframe(df)
results = kb.search("query", k=5)
```

#### `RAGSystem`
```python
from src.models.rag_system import RAGSystem

rag = RAGSystem(df=df, vector_retriever=retriever, memory_manager=memory)
response = rag.query("What is the total sales?")
```

### Key Methods

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `load_data()` | file_path: str | DataFrame | Loads and normalizes CSV |
| `create_from_dataframe()` | df: DataFrame | VectorStore | Creates Chroma index |
| `query()` | question: str | str | Processes natural language query |
| `create_sales_trend()` | date_col, sales_col, period | Figure | Generates time-series chart |
| `search()` | query: str, k: int | List[Document] | Vector similarity search |

## Troubleshooting

### Common Issues

**1. "OpenAI API key not found"**
- Ensure `.env` file exists in project root
- Check format: `OPENAI_API_KEY=sk-...` (no spaces, no quotes)
- Restart Streamlit after creating .env

**2. "Knowledge base creation failed"**
- Verify API key is valid at https://platform.openai.com/api-keys
- Check internet connection
- Ensure sufficient OpenAI credits

**3. "Error loading data"**
- Check CSV file format
- Ensure file has header row
- Verify column names don't have special characters

**4. Slow performance**
- Use `gpt-3.5-turbo` instead of `gpt-4` for faster responses
- Reduce chunk size in knowledge base creation
- Limit data size to < 10,000 rows for optimal performance

**5. Visualizations not showing**
- Ensure required columns exist (Date, Sales, Product, etc.)
- Check column name casing
- Try different visualization types

### Getting Help

1. Check error message in Streamlit UI
2. Review logs in terminal
3. Verify environment variables are loaded
4. Test with sample data first

## Advanced Usage

### Custom Prompts

Modify `src/utils/prompts.py`:

```python
SYSTEM_PROMPT = """
You are a specialized business analyst focusing on [your domain].
[Add custom instructions]
"""
```

### Add New Visualizations

Extend `src/visualization/charts.py`:

```python
def create_custom_chart(self, data_col: str) -> go.Figure:
    # Your visualization logic
    fig = px.scatter(self.df, x=..., y=...)
    return fig
```

### Adjust Retrieval

Configure in `src/models/custom_retriever.py`:

```python
k = 5  # Number of documents to retrieve
score_threshold = 0.7  # Minimum similarity score
```

## Performance Optimization

**For Large Datasets (>10,000 rows)**:
1. Increase chunk size: `chunk_size=2000`
2. Reduce overlap: `chunk_overlap=100`
3. Use batch processing for embeddings
4. Consider data sampling for prototyping

**For Faster Responses**:
1. Use `gpt-3.5-turbo` model
2. Reduce context window: `k=3`
3. Enable caching for repeated queries
4. Pre-compute common statistics

## Security Best Practices

1. **Never commit `.env`** to version control (already in .gitignore)
2. **Rotate API keys** regularly
3. **Sanitize user inputs** before processing
4. **Use environment-specific configs** for dev/prod
5. **Implement rate limiting** for production deployments

## Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository at https://share.streamlit.io
3. Add secrets in dashboard: `OPENAI_API_KEY`
4. Deploy

### Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t insightforge .
docker run -p 8501:8501 --env-file .env insightforge
```

## Project Structure Details

### Data Normalization Strategy

All data is normalized at load time to prevent type issues:
- **Dates** → String format (YYYY-MM-DD)
- **Numbers** → int64 or float64
- **Text** → Clean strings (no NaN)

This ensures PyArrow compatibility and eliminates display errors.

### Vector Database Choice

**Why Chroma over FAISS?**
- ✅ Simpler installation (pure Python)
- ✅ Built-in persistence
- ✅ Better error messages
- ✅ Active development
- ✅ No external dependencies

### Memory Architecture

Custom sliding-window memory using Python's `deque`:
- Stores last K exchanges (default: 5)
- Automatically truncates old messages
- Thread-safe operations
- No external dependencies

## Limitations

1. **Dataset Size**: Optimal for < 100,000 rows
2. **File Format**: CSV only (no Excel, JSON, etc.)
3. **Language**: English queries only
4. **Update Frequency**: Manual re-upload required for data updates
5. **Concurrent Users**: Single-session design (use Streamlit Cloud for multi-user)

## Future Enhancements

Potential improvements:
- Multi-file support (Excel, JSON, databases)
- Real-time data connections
- Custom dashboard builder
- Export to PowerPoint/PDF
- Multi-language support
- Scheduled data refreshes
- User authentication
- Advanced forecasting models

## License

Educational project created for the Advanced Generative AI Capstone program.

## Acknowledgments

- **OpenAI** for GPT models and embeddings
- **LangChain/LangGraph** for RAG framework
- **Chroma** for vector database
- **Streamlit** for UI framework
- **Simplilearn** for the Capstone program

## References

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Version**: 1.0.0
**Last Updated**: February 2026
**Status**: Production Ready
