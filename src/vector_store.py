import os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from typing import List , Dict, Any
from langchain_core.documents import Document
from src.embeddings import TransformerEmbeddingModel
from src.bm25_index import BM25encoder
from src.fusion import rrf
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



class API_KEY_ERROR(Exception):
    pass


class PineconeVectorDB:

    def __init__(self,
                index_name: str,
                namespace: str,
                pinecone_api_key: str):

        self.__index_name = index_name
        self.__namespace = namespace
        self.__index = None

        try:
            self.transformerembeddingmodel = TransformerEmbeddingModel()
            self.bm25encoder = BM25encoder()
            load_dotenv()

            if os.getenv(pinecone_api_key):
                self.__pinecone_client = Pinecone(api_key = os.getenv(pinecone_api_key))
                self.__pinecone_Create_index()

            else:
                logging.error("Their is no api-key ")
                raise API_KEY_ERROR("This is no pinecone_api_key")

        except Exception as e:
            logging.error(f"{e}")
            raise e
        

    def __pinecone_Create_index(self):

        if not self.__pinecone_client.has_index(self.__index_name):

            self.__pinecone_client.create_index(name = self.__index_name,
                                              vector_type = "dense",
                                              dimension=384,
                                              metric="dotproduct",
                                              spec=ServerlessSpec(cloud="aws",
                                                                  region="us-east-1"))
            logging.info(f"{self.__index_name}  have created successfully")
            
        else:
            logging.info(f"{self.__index_name} already have created successfully.")

        if self.__index is None:
            self.__index = self.__pinecone_client.Index(name = self.__index_name)



    def pinecone_upsert_data(self, document: List[Document]):

        try:

            if not isinstance(document[0], Document):
                logging.error("No document data")
                raise ValueError("No document data. Then no process")
            
            dense_vector = self.transformerembeddingmodel.document_embedding(document)
            sparse_vector = self.bm25encoder.bm25_DocumentEncoder(document)

            if len(dense_vector) != len(sparse_vector):
                logging.error(f"dense vector len {len(dense_vector)} != sparse vector len {len(sparse_vector)}")
                raise ValueError(f"dense vector len {len(dense_vector)} != sparse vector len {len(sparse_vector)}.")
            
            records : List[Dict[str,Any]] = []

            for (da , de , se) in zip(document , dense_vector , sparse_vector):

                records.append({
                    "id": da.metadata["chunk_id"],
                    "values" : de,
                    "sparse_values" : {"indices": se["indices"], "values" : se["values"]},
                    "metadata" : {"text" : da.page_content}
                    })
                
            self.__index.upsert(vectors = records , namespace = self.__namespace)
            logging.info("data have index succussfully")

        except Exception as e:
            logging.error(f"{e}")
            raise e



    def pinecone_QuerySearch(self, query: str, 
                             top_k: int = 10, 
                             include_values: bool = False, 
                             include_metadata: bool = True):

        try:

            dense_vector = self.transformerembeddingmodel.query_embedding(query)
            sparse_vector = self.bm25encoder.bm25_QueryEncoder(query)

            logging.info("query text have convert into dense and sparse vectors")


            dense_retrieve = self.__index.query(
                namespace = self.__namespace,
                top_k = top_k,
                vector = dense_vector,
                include_metadata = include_metadata,
                include_values = include_values
            )

            logging.info("dense vector document have retrieved succussfully")


            dump_dense_vector = [0] * 384

            sparse_retrieve = self.__index.query(
                namespace = self.__namespace,
                top_k = top_k,
                vector = dump_dense_vector,
                sparse_vector = {"indices":sparse_vector["indices"] , "values":sparse_vector["values"]},
                include_values = include_values,
                include_metadata = include_metadata
            )

            logging.info("sparse vector document have retrieved succussfully")

            both_vector_retrieve : List[List[Any]] = []

            d = [{"id" : d.id,"text":d.metadata["text"]} 
                 for d in dense_retrieve.matches] 
            both_vector_retrieve.append(d)

            s = [{"id" : s.id, "score": s.score,"text":s.metadata["text"]} 
                 for s in sparse_retrieve.matches]
            both_vector_retrieve.append(s)
            
            return both_vector_retrieve
            

        except Exception as e:
            logging.error(f"{e}")
            raise e