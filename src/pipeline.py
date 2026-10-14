from src.document_loader import DocumentLoader
from src.chunking import TextChunking
from src.embeddings import TransformerEmbeddingModel
from src.fusion import rrf 
from src.generation import LLM
from src.reranking import RerankingModel
from src.vector_store import PineconeVectorDB
from src.bm25_index import BM25encoder


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



class DataIndexing:

    def __init__(self):


        try:
            self.document_loader = DocumentLoader
            self.chunking = TextChunking()
            self.embedding_model = TransformerEmbeddingModel
            self.vector_indexing = PineconeVectorDB
            self.bm25 = BM25encoder
            self.rrf  = rrf
            self.reranking = RerankingModel

        except Exception as e:
            logging.error(f"{e} in class loading in data indexing")
            raise e


    def data_Indexing(self,
                 pdf_path: str,
                 bm25_data_store : str, 
                 vector_db_index_name : str,
                 vector_db_namespace : str,
                 chunk_size: int = 500, 
                 chunk_overlap: int = 150, 
                 separators: list[str] = ["\n\n", "\n", ". ", " ",""],
                 batch_size: int = 200):

        # pdf document loader
        it_parse_data = self.document_loader(pdf_path).file_path()
        logging.info("parsing data done ✔️")

        it_chunk = self.chunking.recursive_CharacterTextSplitter(it_parse_data,
                                                              chunks_size = chunk_size,
                                                              chunks_over_lap = chunk_overlap,
                                                              separators = separators)
        logging.info("data have chunked ✔️")

        '''is_embedding_model = self.embedding_model().document_embedding(it_chunk,batch_size)
        logging.info(f"data have convert into dense vector by {self.embedding_model().model_name} embedding model")

        is_bm25 = self.bm25(bm25_data_store).bm25_DocumentEncoder(it_chunk)'''

        vector_db_indexing = self.vector_indexing(index_name = vector_db_index_name,
                                                  namespace = vector_db_namespace).pinecone_upsert_data(it_chunk)

        logging.info("vector db indexing have done successfully ✔️")


class DataRetriving:
    
    def __init__(self):

        try:

                self.llm_model = LLM
                self.embedding_model = TransformerEmbeddingModel
                self.vector_indexing = PineconeVectorDB
                self.bm25 = BM25encoder
                self.rrf  = rrf
                self.reranking = RerankingModel
    
        except Exception as e:
                logging.error(f"{e} in class loading in data indexing")
                raise e


        
    def data_retriving(self, query: str,
                       vector_index_name: str,
                       vector_namespace: str,
                       top_K: int,
                       include_values: bool = False,
                       include_metadata: bool = True,
                       reranking_model: str = "rerank-v4.0-pro",
                       llm_model: str = "openai/gpt-oss-120b"):

        try:
            
            vector_db_retrieveing = self.vector_indexing(index_name=vector_index_name ,namespace=vector_namespace).pinecone_QuerySearch(query=query,
                                                                                                                                    top_k=top_K,
                                                                                                                                    include_values=include_values,
                                                                                                                                    include_metadata=include_metadata)
            logging.info("successfully retreive the data from vector-db")
            is_rrf = self.rrf(vectors_result=vector_db_retrieveing)
            logging.info("successfully rank the document by rrf algo")
             
            is_reranking = self.reranking(model_name=reranking_model).reranker(query=query,
                                                                                  document=is_rrf)
            logging.info("reranking model have ranked the documents")
             
            llm = self.llm_model(model_name=llm_model).llm_Generator(query=query,
                                                                              document=is_reranking)
            logging.info("successfully llm have generate the finall answer")
             
            return llm

        except Exception as e:
             logging.error(e)
             raise e
    