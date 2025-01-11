"""
Module for computing sentence embeddings using BERT.
"""

import torch
from transformers import BertModel, BertTokenizer
from typing import Optional
from pathlib import Path

class BertEmbedder:
    """
    A class to compute sentence embeddings using BERT.

    Parameters
    ----------
    model_name : str, optional
        Name of the pre-trained BERT model to use.
    cache_dir : Path, optional
        Directory to cache the model.
    device : str, optional
        Device to run the model on ('cpu' or 'cuda').
    """

    def __init__(
        self, 
        model_name: str = 'bert-base-uncased',
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.model = BertModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()

    def get_sentence_embedding(
        self, 
        sentence: str,
        max_length: int = 128
    ) -> torch.Tensor:
        """
        Computes the embedding for a given sentence.

        Parameters
        ----------
        sentence : str
            The input sentence.
        max_length : int, optional
            Maximum sequence length.

        Returns
        -------
        torch.Tensor
            The embedding vector for the sentence.

        Notes
        -----
        The embedding is computed by taking the mean of the last hidden states
        of the BERT model over all tokens.
        """
        # Tokenize input
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Compute embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get mean pooled output
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        
        # Move back to CPU if needed
        if self.device != 'cpu':
            embedding = embedding.cpu()
            
        return embedding

    def __call__(self, sentence: str) -> torch.Tensor:
        """
        Callable interface for the embedder.

        Parameters
        ----------
        sentence : str
            The input sentence.

        Returns
        -------
        torch.Tensor
            The embedding vector for the sentence.
        """
        return self.get_sentence_embedding(sentence)
