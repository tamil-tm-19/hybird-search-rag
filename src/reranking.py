import os
import cohere
from typing import List


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
            raise ValueError("cohere api-key not found in .env file. Please set the cohere reranker api-key")
        except Exception as e:
            print(f"ERROR : {e}")


    def reranker(self, query: str, document: List[str], top_n: int = 3)-> List[str]:

        try:
            if not self.cohere_reranking_model:
                raise ReRankerModelNotFounded("re-ranking model not loaded.")
            
            reranking_result = self.cohere_reranking_model.rerank(model = self.model_name, 
                                                                  query = query, 
                                                                  documents = document, 
                                                                  top_n = top_n)
            finall_result = [document[result.index] 
                             for result in reranking_result.results
                             if result]
            return finall_result

        except Exception as e:
            print(f"ERROR : {e}")