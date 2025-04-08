# Agentic AI Data Analyst

## 🌟 Overview

Agentic AI Data Analyst is an advanced, AI-powered data analysis and visualization platform that leverages the capabilities of large language models to provide intelligent data insights. This application enables users to interact with their data through natural language, perform complex data operations, and generate visualizations without writing code.

## 🚀 Features

### Natural Language Data Analysis
- Ask questions about your data in plain English
- Generate visualizations by describing what you want to see
- Receive explanations and insights in conversational format

### Multi-Format Data Support
- CSV, Excel, and Parquet file processing
- Support for multiple files with seamless switching
- Advanced DataFrame operations (merge, join, concat, groupby, etc.)

### Data Quality & Enhancement
- Comprehensive data quality reporting
- Duplicate record detection with fuzzy matching
- Missing value analysis and handling

### PDF Document Analysis
- Direct PDF downloads from URLs
- Annual report retrieval and analysis
- Question answering based on document content

### Advanced Visualization
- Automated chart generation based on data characteristics
- Interactive data exploration
- Export capabilities for reports and findings

## 🔧 Architecture

The application is built with a modular architecture:

```
project_root/
├── config/              # Configuration files
│   ├── __init__.py
│   ├── streamlit_config.py
│   └── python_engine.py
├── docs/                # Documentation
│   ├── README.md
│   └── requirement.txt
├── src/                 # Source code
│   ├── __init__.py
│   ├── data_operation.py      # DataFrame operations
│   ├── duplicate_detector.py  # Fuzzy duplicate detection
│   ├── file_export.py         # File export utilities
│   ├── llm.py                 # LLM integration
│   ├── pdf_fetcher.py         # PDF retrieval
│   └── rag_app.py             # Retrieval-augmented generation
├── tests/               # Test files
│   ├── __init__.py
│   └── test_duplicate_detector.py
└── main.py              # Application entry point
```

### Key Components:

- **Streamlit Interface**: Provides the web UI for user interaction
- **OpenAI Integration**: Powers the natural language understanding and code generation
- **Data Processing Engine**: Handles complex data operations and transformations
- **RAG System**: Enables question answering on PDF documents
- **Visualization Generator**: Creates data visualizations based on context

## 📋 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/agentic-ai-data-analyst.git
   cd agentic-ai-data-analyst
   ```

2. Install dependencies:
   ```bash
   pip install -r docs/requirement.txt
   ```

3. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Or provide it through the application interface.

4. Run the application:
   ```bash
   streamlit run main.py
   ```

## 💻 Usage

### Basic Workflow

1. **Load Data**: Upload CSV, Excel, or Parquet files through the interface
2. **Ask Questions**: Type natural language queries about your data
3. **Perform Operations**: Request data transformations or conduct specialized analysis
4. **Generate Visualizations**: Ask for specific charts or let the AI suggest appropriate visualizations
5. **Export Results**: Save analysis results or processed data to your preferred format

### Example Queries

- "Show me a histogram of sales by region"
- "Find duplicate customer records"
- "Merge these two files on customer_id"
- "Check the data quality of this dataset"
- "Download the annual report for Apple and tell me about their revenue growth"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- OpenAI for GPT models
- Streamlit for the interactive web framework
- Pandas, NumPy, and Matplotlib for data processing and visualization
- LangChain for RAG capabilities

## 📞 Contact

For questions or feedback, please open an issue on GitHub or contact the maintainers mail : omkarsatapathy001@gmail.com.

---

**Note**: This application requires an OpenAI API key and may incur API usage costs. Always review OpenAI's pricing and terms before extended use.