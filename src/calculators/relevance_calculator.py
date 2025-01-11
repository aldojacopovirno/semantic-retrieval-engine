"""
Module for calculating overall document relevance scores.
"""
from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class RelevanceMetrics:
    """Container for document relevance metrics."""
    filename: str
    relevance_score: float
    similarity_score: float
    tfidf_score: float
    keyword_count: int
    keyword_percentage: float
    avg_keyword_position: float

class RelevanceCalculator:
    """
    A class to calculate the relevance of documents based on multiple criteria.
    
    Parameters
    ----------
    documents : list of str
        The list of documents.
    filenames : list of str
        The filenames corresponding to the documents.
    weights : dict, optional
        Weights for different components of the relevance score.
    """
    def __init__(
        self,
        documents: List[str],
        filenames: List[str],
        weights: dict = None
    ):
        if len(documents) != len(filenames):
            raise ValueError("Number of documents must match number of filenames")
        
        self.documents = documents
        self.filenames = filenames
        self.weights = weights or {
            'similarity': 0.4,
            'tfidf': 0.3,
            'keyword_occurrence': 0.2,
            'position': 0.1
        }
        
        # Validate weights
        if sum(self.weights.values()) != 1.0:
            logger.warning("Weights do not sum to 1.0, normalizing...")
            total = sum(self.weights.values())
            self.weights = {k: v/total for k, v in self.weights.items()}

    def _calculate_keyword_metrics(self, document: str, keyword: str) -> Tuple[int, float, float]:
        """
        Calculate keyword-related metrics for a single document.
        
        Parameters
        ----------
        document : str
            The document text to analyze.
        keyword : str
            The keyword to search for.
            
        Returns
        -------
        Tuple[int, float, float]
            Returns (keyword_count, percentage_occurrence, avg_position)
        """
        try:
            words = document.lower().split()
            word_count = len(words)
            
            if word_count == 0:
                logger.warning("Empty document found")
                return 0, 0.0, -1.0
            
            # Find keyword occurrences and positions
            keyword = keyword.lower()
            keyword_positions = [i for i, word in enumerate(words) if word == keyword]
            keyword_count = len(keyword_positions)
            
            # Calculate percentage and average position
            percentage = (keyword_count / word_count) * 100
            avg_position = (
                sum(keyword_positions) / keyword_count 
                if keyword_count > 0 
                else -1.0
            )
            
            return keyword_count, percentage, avg_position
            
        except Exception as e:
            logger.error(f"Error calculating keyword metrics: {str(e)}")
            return 0, 0.0, -1.0

    def calculate_relevance(
        self,
        cosine_similarities: List[float],
        tfidf_scores: List[float],
        keyword: str
    ) -> List[RelevanceMetrics]:
        """
        Calculates the relevance score for each document.
        
        Parameters
        ----------
        cosine_similarities : list of float
            Cosine similarity scores for the documents.
        tfidf_scores : list of float
            TF-IDF scores for the documents.
        keyword : str
            The keyword used for calculating relevance.
            
        Returns
        -------
        list of RelevanceMetrics
            Relevance metrics for each document.
        """
        if not all(len(lst) == len(self.documents) for lst in [cosine_similarities, tfidf_scores]):
            raise ValueError("Length of similarity and TF-IDF scores must match number of documents")

        results = []
        
        for idx, (doc, filename) in enumerate(zip(self.documents, self.filenames)):
            try:
                # Calculate keyword metrics
                keyword_count, keyword_percentage, avg_position = self._calculate_keyword_metrics(doc, keyword)
                
                # Normalize position score (0 to 1, where 1 is better - appearing earlier in document)
                position_score = (
                    1 - (avg_position / len(doc.split())) 
                    if avg_position >= 0 
                    else 0
                )
                
                # Calculate weighted relevance score
                relevance_score = (
                    self.weights['similarity'] * cosine_similarities[idx] +
                    self.weights['tfidf'] * tfidf_scores[idx] +
                    self.weights['keyword_occurrence'] * (keyword_percentage / 100) +
                    self.weights['position'] * position_score
                )
                
                # Create metrics object
                metrics = RelevanceMetrics(
                    filename=filename,
                    relevance_score=float(relevance_score),
                    similarity_score=float(cosine_similarities[idx]),
                    tfidf_score=float(tfidf_scores[idx]),
                    keyword_count=keyword_count,
                    keyword_percentage=float(keyword_percentage),
                    avg_keyword_position=float(avg_position)
                )
                
                results.append(metrics)
                
            except Exception as e:
                logger.error(f"Error calculating relevance for document {filename}: {str(e)}")
                # Add a zero-scored result in case of error
                results.append(RelevanceMetrics(
                    filename=filename,
                    relevance_score=0.0,
                    similarity_score=0.0,
                    tfidf_score=0.0,
                    keyword_count=0,
                    keyword_percentage=0.0,
                    avg_keyword_position=-1.0
                ))
        
        return results
