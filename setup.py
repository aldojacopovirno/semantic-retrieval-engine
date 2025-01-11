from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="semantic-retrieval-engine",
    version="0.1.0",
    author="Aldo Jacopo Virno, Andrea Bucchignani",
    author_email="aldojacopo@gmail.com, andreabucchignani@gmail.com",
    description="A hybrid document search engine combining BERT embeddings and TF-IDF for intelligent semantic retrieval and ranking of documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aldojacopovirno/SemanticRetrievalEngine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: General",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Information Retrieval",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Office Suites",
        "Topic :: Database :: Database Engines/Servers",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.5.0",
        "scikit-learn>=0.24.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "semantic-search=semantic_retrieval_engine.main:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/SemanticRetrievalEngine/issues",
        "Source": "https://github.com/yourusername/SemanticRetrievalEngine",
    },
    license="Apache License 2.0",
)
