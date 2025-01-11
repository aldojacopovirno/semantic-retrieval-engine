"""
Module for loading and preprocessing text documents.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    A class to load and preprocess text documents from a folder.

    Parameters
    ----------
    folder_path : str or Path
        The path to the folder containing text documents.
    encoding : str, optional
        The encoding to use when reading files.
    file_extension : str, optional
        The file extension to filter for.

    Attributes
    ----------
    documents : list of str
        The loaded text documents.
    filenames : list of str
        The filenames of the loaded documents.
    """

    def __init__(
        self,
        folder_path: str | Path,
        encoding: str = 'utf-8',
        file_extension: str = '.txt'
    ):
        self.folder_path = Path(folder_path)
        self.encoding = encoding
        self.file_extension = file_extension
        self.documents: List[str] = []
        self.filenames: List[str] = []

        if not self.folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

    def load_documents(self) -> None:
        """
        Loads all text files from the specified folder.

        Returns
        -------
        None

        Raises
        ------
        IOError
            If there are problems reading the files.
        """
        try:
            for file_path in self.folder_path.glob(f"*{self.file_extension}"):
                try:
                    with open(file_path, 'r', encoding=self.encoding) as file:
                        text = file.read()
                        # Basic preprocessing
                        text = self._preprocess_text(text)
                        
                        self.documents.append(text)
                        self.filenames.append(file_path.name)
                except UnicodeDecodeError:
                    logger.warning(f"Failed to decode file {file_path} with {self.encoding} encoding")
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")

            if not self.documents:
                logger.warning(f"No {self.file_extension} files found in {self.folder_path}")

        except Exception as e:
            raise IOError(f"Error loading documents: {str(e)}")

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocesses the input text.

        Parameters
        ----------
        text : str
            The input text to preprocess.

        Returns
        -------
        str
            The preprocessed text.
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text

    def get_document(self, index: int) -> Tuple[str, str]:
        """
        Retrieves a specific document and its filename.

        Parameters
        ----------
        index : int
            The index of the document to retrieve.

        Returns
        -------
        tuple of (str, str)
            The document text and filename.

        Raises
        ------
        IndexError
            If the index is out of range.
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")
        return self.documents[index], self.filenames[index]

    def __len__(self) -> int:
        """Returns the number of loaded documents."""
        return len(self.documents)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        """Allows indexing to get documents."""
        return self.get_document(index)
