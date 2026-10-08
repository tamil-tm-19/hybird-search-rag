import os
import cohere
from typing import List
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


class ReRankerModelNotFounded(Exception):
    pass

class RerankingModel:

    def __init__(self, 
                 api_key: str,
                 model_name: str = "rerank-v4.0-pro"):

        self.model_name = model_name

        try:
            get_api_key = os.getenv(api_key)

            if get_api_key:
                self.cohere_reranking_model = cohere.ClientV2(api_key=get_api_key)
                logging.info("re-ranking model have loaded")

            else:
                logging.error("cohere api-key not found in .env file. Please set the cohere reranker api-key")
                raise ValueError("cohere api-key not found in .env file. Please set the cohere reranker api-key")
        except Exception as e:
            logging.error(f"ERROR : {e}")
            raise e


    def reranker(self, query: str, document: List[str], top_n: int = 3)-> List[str]:

        try:
            if not self.cohere_reranking_model:
                logging.error("re-ranking model not loaded.")
                raise ReRankerModelNotFounded("re-ranking model not loaded.")
            
            reranking_result = self.cohere_reranking_model.rerank(model = self.model_name, 
                                                                  query = query, 
                                                                  documents = document, 
                                                                  top_n = top_n)

            logging.info("re-ranker model is ranking the document....")
            finall_result = [document[result.index] 
                             for result in reranking_result.results
                             if result]
            logging.info("ranked the document")
            return finall_result

        except Exception as e:
            logging.error(f"ERROR : {e}")
            raise e