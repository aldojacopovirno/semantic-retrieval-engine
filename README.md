# Semantic Retrieval Engine: A Multi-Feature Model for Contextual and Keyword-Aware Search

A Python-based semantic search engine that combines BERT (Bidirectional Encoder Representations from Transformers) embeddings and TF-IDF (Term Frequency-Inverse Document Frequency) for intelligent document retrieval and ranking. This system leverages advanced Natural Language Processing techniques to provide highly accurate and context-aware document search capabilities.

## Features

- BERT-based semantic document embeddings
- TF-IDF scoring for keyword relevance
- Hybrid ranking system combining multiple relevance signals
- Detailed search results with metrics

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

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{SemanticRetrievalEngine,
  title = {Semantic Retrieval Engine: A Multi-Feature Model for Contextual and Keyword-Aware Search},
  author = {Aldo Jacopo Virno, Andrea Bucchignani},
  year = {2025},
  url = {https://github.com/aldojacopovirno/SemanticRetrievalEngine}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
