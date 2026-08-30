import os
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from typing import List , Dict, Any
from langchain_core.documents import Document
from src.embeddings import TransformerEmbeddingModel
from src.bm25_index import BM25encoder
from src.fusion import rrf



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
                raise API_KEY_ERROR("this is no pinecone_api_key")

        except Exception as e:
            print(f"ERROR : {e}")
            raise e
        

    def __pinecone_Create_index(self):

        if not self.__pinecone_client.has_index(self.__index_name):
            self.__pinecone_client.create_index(name = self.__index_name,
                                              vector_type = "dense",
                                              dimension=384,
                                              metric="dotproduct",
                                              spec=ServerlessSpec(cloud="aws",
                                                                  region="us-east-1"))
            print(f"{self.__index_name}  have created successfully")
            
        else:
            print(f"{self.__index_name} already have created successfully.")

        if self.__index is None:
            self.__index = self.__pinecone_client.Index(name = self.__index_name)



    def pinecone_Upsert_data(self, document: List[Document]):

        try:
            if not document:
                raise ValueError("No document No process")
            
            if not self.__namespace:
                raise ValueError("give some namespace to store in your index.")
            
            if not isinstance(document[0], Document):
                raise ValueError("No document data. Then no process")
            
            dense_vector = self.transformerembeddingmodel.document_embedding(document)
            sparse_vector = self.bm25encoder.bm25_DocumentEncoder(document)

            if len(dense_vector) != len(sparse_vector):
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

        except Exception as e:
            print(f"ERROR : {e}")
            raise e



    def pinecone_QuerySearch(self, query: str, top_k: int = 10, include_values: bool = False, include_metadata: bool = True , rrf_topk: int = 5):

        try:

            dense_vector = self.transformerembeddingmodel.query_embedding(query)
            sparse_vector = self.bm25encoder.bm25_QueryEncoder(query)

            dense_retrieve = self.__index.query(
                namespace = self.__namespace,
                top_k = top_k,
                vector = dense_vector,
                include_metadata = include_metadata,
                include_values = include_values
            )

            dump_dense_vector = [0] * 384

            sparse_retrieve = self.__index.query(
                namespace = self.__namespace,
                top_k = top_k,
                vector = dump_dense_vector,
                sparse_vector = {"indices":sparse_vector["indices"] , "values":sparse_vector["values"]},
                include_values = include_values,
                include_metadata = include_metadata
            )

            both_retrieve : List[List[Any]] = []

            d = [{"id" : d.id,"text":d.metadata["text"]} 
                 for d in dense_retrieve.matches] 
            both_retrieve.append(d)

            s = [{"id" : s.id, "score": s.score,"text":s.metadata["text"]} 
                 for s in sparse_retrieve.matches]
            both_retrieve.append(s)

            # RRF
            rrf_result = rrf(both_retrieve , rrf_topk)
            return rrf_result
            

        except Exception as e:
            print(f"ERROR : {e}")
            raise e