# Semantic Retrieval Engine

A Python-based semantic search engine that uses BERT embeddings and TF-IDF to find relevant documents based on natural language queries.

## Features

- BERT-based semantic document embeddings
- TF-IDF scoring for keyword relevance
- Hybrid ranking system combining multiple relevance signals
- Detailed search results with various metrics
- Easy-to-use command line interface

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

## Documentation

See the [docs](docs/) directory for detailed documentation:
- [API Documentation](docs/API.md)
- [Usage Guide](docs/USAGE.md)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
