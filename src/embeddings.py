import torch
from typing import List, Optional
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer





class EmbeddingModelNotFound(Exception):
    pass

class NoDocumentInList(Exception):
    pass




class TransformerEmbeddingModel:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.embedding_model = SentenceTransformer(model_name , device = self.device)

        except Exception as e:
            raise EmbeddingModelNotFound(f"{model_name} Embedding model not loaded. {e}")




    @staticmethod
    def document_Cleaning(document: List[Document])-> List[str]:

        try:
            if isinstance(document[0] , Document):
                data_cleaning = [doc.page_content.strip() for doc in document]
                return data_cleaning
            raise NoDocumentInList(f"no document in list.")

        except Exception as e:
            print(f"ERROR : {e}")
            raise e




    def document_embedding(self, document: List[Document], batch_size: int | None = None)-> List[List[float]]:

        if isinstance(document[0] , Document):
            clean_text = self.document_Cleaning(document)

        if batch_size is None:
            batch_size = 200

        try:

            if self.embedding_model:
                embedding_document = self.embedding_model.encode(
                    clean_text,
                    batch_size= batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True).tolist()

                return embedding_document

            raise EmbeddingModelNotFound(f"{self.model_name} Embedding model not loaded.")

        except Exception as e:
            print(f"ERROR : {e}")
            raise e


    def query_embedding(self, query: str | list[str])-> List[str]:

        try:
            if self.embedding_model:
                embedding_query = self.embedding_model.encode(
                    query,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True).tolist()
                return embedding_query

            raise EmbeddingModelNotFound(f"{self.model_name} Embedding model not loaded.")
        
        except Exception as e:
            print(f"ERROR : {e}")
            raise e