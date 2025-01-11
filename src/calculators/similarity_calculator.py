"""
Module for calculating semantic similarity between documents.
"""

from typing import List
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from ..embedders.bert_embedder import BertEmbedder

logger = logging.getLogger(__name__)

class SimilarityCalculator:
    """
    A class to calculate cosine similarity between query and documents.

    Parameters
    ----------
    embedder : BertEmbedder
        An instance of the BertEmbedder class.
    documents : list of str
        The list of documents to calculate similarity for.
    batch_size : int, optional
        Batch size for processing documents.
    """

    def __init__(
        self,
        embedder: BertEmbedder,
        documents: List[str],
        batch_size: int = 32
    ):
        self.embedder = embedder
        self.documents = documents
        self.batch_size = batch_size
        self.document_embeddings: List[torch.Tensor] = []

    def calculate_document_embeddings(self) -> None:
        """
        Computes embeddings for all documents in batches.
        """
        try:
            self.document_embeddings = []
            for i in range(0, len(self.documents), self.batch_size):
                batch = self.documents[i:i + self.batch_size]
                batch_embeddings = [
                    self.embedder.get_sentence_embedding(doc)
                    for doc in batch
                ]
                self.document_embeddings.extend(batch_embeddings)
            
            logger.info(f"Calculated embeddings for {len(self.documents)} documents")
        
        except Exception as e:
            logger.error(f"Error calculating document embeddings: {str(e)}")
            raise

    def calculate_cosine_similarity(self, query: str) -> List[float]:
        """
        Computes cosine similarity between the query and all documents.

        Parameters
        ----------
        query : str
            The query to compare with the documents.

        Returns
        -------
        list of float
            Cosine similarity scores for each document.
        """
        try:
            if not self.document_embeddings:
                logger.warning("Document embeddings not calculated. Running calculation now.")
                self.calculate_document_embeddings()

            query_embedding = self.embedder.get_sentence_embedding(query)
            
            similarities = [
                float(cosine_similarity(
                    query_embedding.reshape(1, -1),
                    doc_embedding.reshape(1, -1)
                )[0, 0])
                for doc_embedding in self.document_embeddings
            ]
            
            return similarities
        
        except Exception as e:
            logger.error(f"Error calculating similarities: {str(e)}")
            return [0.0] * len(self.documents)
