# Semantic Retrieval Engine

A Python-based semantic search engine that uses BERT embeddings and TF-IDF to find relevant documents based on natural language queries.

## Features

- BERT-based semantic document embeddings
- TF-IDF scoring for keyword relevance
- Hybrid ranking system combining multiple relevance signals
- Detailed search results with various metrics
- Easy-to-use command line interface

## Repository Structure

```bash
semantic_retrieval_engine/
├── src/
│   ├── __init__.py
│   ├── embedders/
│   │   ├── __init__.py
│   │   └── bert_embedder.py
│   ├── loaders/
│   │   ├── __init__.py
│   │   └── document_loader.py
│   ├── scorers/
│   │   ├── __init__.py
│   │   └── tfidf_scorer.py
│   ├── calculators/
│   │   ├── __init__.py
│   │   ├── similarity_calculator.py
│   │   └── relevance_calculator.py
│   ├── displayers/
│   │   ├── __init__.py
│   │   └── results_displayer.py
│   └── main.py
├── requirements.txt
├── setup.py
├── README.md
├── LICENSE
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from semantic_retrieval_engine.src.main import main

# Run the search engine
main()
```

Or use the command line:

```bash
python -m semantic_retrieval_engine.src.main
```

## Usage

1. Place your text documents in a folder
2. Update the folder path in the configuration
3. Run the search engine
4. Enter your search query
5. View results in the console and in the generated report

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- scikit-learn
- Other dependencies listed in requirements.txt

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
