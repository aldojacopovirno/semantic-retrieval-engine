"""
Main module for the Semantic Retrieval Engine.
"""

import os
import argparse
from typing import Optional

from semantic_retrieval_engine.src.embedders.bert_embedder import BertEmbedder
from semantic_retrieval_engine.src.loaders.document_loader import DocumentLoader
from semantic_retrieval_engine.src.scorers.tfidf_scorer import TFIDFScorer
from semantic_retrieval_engine.src.calculators.similarity_calculator import SimilarityCalculator
from semantic_retrieval_engine.src.calculators.relevance_calculator import RelevanceCalculator
from semantic_retrieval_engine.src.displayers.results_displayer import ResultsDisplayer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Semantic Retrieval Engine')
    parser.add_argument('--folder', type=str, help='Path to the documents folder')
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                       help='BERT model name')
    return parser.parse_args()

def main(folder_path: Optional[str] = None, query: Optional[str] = None, 
         model_name: str = 'bert-base-uncased'):
    """
    Main function to run the semantic search engine.

    Parameters
    ----------
    folder_path : str, optional
        Path to the documents folder. If None, will prompt user.
    query : str, optional
        Search query. If None, will prompt user.
    model_name : str, optional
        Name of the BERT model to use.
    """
    args = parse_arguments()
    
    # Use command line arguments if provided
    folder_path = args.folder or folder_path
    query = args.query or query
    model_name = args.model or model_name

    # If still no folder path, use default or prompt
    if not folder_path:
        folder_path = os.getenv('SEMANTIC_RETRIEVAL_DOCS', None)
        if not folder_path:
            folder_path = input("Enter the path to documents folder: ").strip()

    # If still no query, prompt user
    if not query:
        query = input("Enter a query to search the documents: ").strip()

    # Initialize components
    loader = DocumentLoader(folder_path)
    loader.load_documents()

    embedder = BertEmbedder(model_name)
    similarity_calculator = SimilarityCalculator(embedder, loader.documents)
    similarity_calculator.calculate_document_embeddings()
    cosine_similarities = similarity_calculator.calculate_cosine_similarity(query)

    keyword = query.split()[0].lower()
    tfidf_scorer = TFIDFScorer(loader.documents)
    tfidf_scores = tfidf_scorer.get_scores(keyword)

    relevance_calculator = RelevanceCalculator(loader.documents, loader.filenames)
    relevance_info = relevance_calculator.calculate_relevance(
        cosine_similarities, tfidf_scores, keyword)

    displayer = ResultsDisplayer()
    displayer.display_results(relevance_info, keyword, query)

if __name__ == '__main__':
    main()
