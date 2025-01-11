"""
Module for TF-IDF based document scoring.
"""

from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)

class TFIDFScorer:
    """
    A class to calculate TF-IDF scores for documents.

    Parameters
    ----------
    documents : list of str
        The list of documents to compute TF-IDF on.
    max_features : int, optional
        Maximum number of features (terms) to consider.
    stop_words : str or list, optional
        Stop words to use ('english' or custom list).
    
    Attributes
    ----------
    vectorizer : TfidfVectorizer
        The TF-IDF vectorizer instance.
    tfidf_matrix : scipy.sparse.csr.csr_matrix
        The TF-IDF matrix for the documents.
    """

    def __init__(
        self,
        documents: List[str],
        max_features: Optional[int] = None,
        stop_words: Optional[str | List[str]] = 'english'
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words=stop_words
        )
        try:
            self.tfidf_matrix: csr_matrix = self.vectorizer.fit_transform(documents)
            logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        except Exception as e:
            logger.error(f"Error computing TF-IDF matrix: {str(e)}")
            raise

    def get_scores(self, keyword: str) -> np.ndarray:
        """
        Computes the TF-IDF scores for a specific keyword.

        Parameters
        ----------
        keyword : str
            The keyword to calculate TF-IDF scores for.

        Returns
        -------
        numpy.ndarray
            The TF-IDF scores for the keyword across all documents.
        """
        try:
            keyword = keyword.lower()
            keyword_index = self.vectorizer.vocabulary_.get(keyword)
            
            if keyword_index is None:
                logger.warning(f"Keyword '{keyword}' not found in vocabulary")
                return np.zeros(self.tfidf_matrix.shape[0])
            
            return self.tfidf_matrix[:, keyword_index].toarray().flatten()
        
        except Exception as e:
            logger.error(f"Error computing scores for keyword '{keyword}': {str(e)}")
            return np.zeros(self.tfidf_matrix.shape[0])

    def get_top_terms(self, n: int = 10) -> List[tuple]:
        """
        Gets the top n terms with highest TF-IDF scores across all documents.

        Parameters
        ----------
        n : int, optional
            Number of top terms to retrieve.

        Returns
        -------
        list of tuple
            List of (term, score) tuples for top terms.
        """
        try:
            scores = np.asarray(self.tfidf_matrix.sum(axis=0)).flatten()
            terms = self.vectorizer.get_feature_names_out()
            
            top_indices = scores.argsort()[-n:][::-1]
            return [(terms[i], scores[i]) for i in top_indices]
        
        except Exception as e:
            logger.error(f"Error getting top terms: {str(e)}")
            return []

    def get_document_vector(self, document_index: int) -> np.ndarray:
        """
        Gets the TF-IDF vector for a specific document.

        Parameters
        ----------
        document_index : int
            Index of the document.

        Returns
        -------
        numpy.ndarray
            TF-IDF vector for the document.
        """
        return self.tfidf_matrix[document_index].toarray().flatten()
