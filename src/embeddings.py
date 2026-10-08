import torch
from typing import List, Optional
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import logging

# level of logger

# DUBUG
# INFO
# WARING
# ERROR
# CRITICAL

logging.basicConfig(filename="log_tracker.text",
                    filemode= "a",
                    level = logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%y-%m-%d %H %M %S")





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
            logging.info(f"Embedding Model Loaded Successfully -> {model_name}")

        except Exception as e:
            logging.error(f"{model_name} Embedding model not loaded. {e}")
            raise EmbeddingModelNotFound(f"{model_name} Embedding model not loaded. {e}")




    @staticmethod
    def document_Cleaning(document: List[Document])-> List[str]:

        try:
            if isinstance(document[0] , Document):
                data_cleaning = [doc.page_content.strip() for doc in document]
                logging.info("document have clean before embedding")
                return data_cleaning

            logging.error(f"no document in list.")
            raise NoDocumentInList(f"no document in list.")

        except Exception as e:
            logging.error(f"ERROR : {e}")
            raise e




    def document_embedding(self, document: List[Document], batch_size: int | None = None)-> List[List[float]]:

        if isinstance(document[0] , Document):
            clean_text = self.document_Cleaning(document)

        if batch_size is None:
            batch_size = 200

        try:

            logging.info("after cleaning converting text to dense")
            if self.embedding_model:
                embedding_document = self.embedding_model.encode(
                    clean_text,
                    batch_size= batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True).tolist()

                logging.info("converting text to dense vector is done")
                

                return embedding_document

            logging.error(f"{self.model_name} Embedding model not loaded")
            raise EmbeddingModelNotFound(f"{self.model_name} Embedding model not loaded")

        except Exception as e:
            logging.error(f"ERROR : {e}")
            raise e


    def query_embedding(self, query: str | list[str])-> List[str]:

        try:
            if self.embedding_model:
                embedding_query = self.embedding_model.encode(
                    query,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True).tolist()

                logging.info("convert the user query into dense vector")
                
                return embedding_query

            logging.error(f"{self.model_name} Embedding model not loaded.")
            raise EmbeddingModelNotFound(f"{self.model_name} Embedding model not loaded.")
        
        except Exception as e:
            logging.error(f"ERROR : {e}")
            raise e