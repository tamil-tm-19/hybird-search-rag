import os
from typing import List, Dict, Any, Union
from pinecone_text.sparse import BM25Encoder
from langchain_core.documents import Document
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


class BM25encoder:

    def __init__(self, bm25_data_store: str):

        try:
            self.bm25encoder = BM25Encoder()
            self.bm25_store = bm25_data_store
            self.bm25_corpus = False

            if os.path.exists(bm25_data_store):
                self.bm25encoder.load(path = bm25_data_store)
                self.bm25loaded = True
                logging.info(f"bm25 encoder have loaded in {bm25_data_store}")
            else:

                logging.info("bm25 encoder not have loaded")
                self.bm25loaded = False

        except Exception as e:
            logging.error(f"ERROR : {e}")
            raise e
        

    @staticmethod
    def clean_data(document: List[Document])-> List[str]:

        try:
            if isinstance(document[0], Document):
                text_cleaning = [doc.page_content.lower().strip() for doc in document]
                logging.info("data have cleaned before encoding text into sparse vector")
                return text_cleaning

        except Exception as e:
            logging.error(f"ERROR : {e}")
            raise e

        

    def bm25_Fit(self, document: List[Document]):

        try:
            if not isinstance(document[0], Document):
                logging.error("document data should be langchain Document")
                raise ValueError("document data should be langchain Document")
            
            clean_document = self.clean_data(document)
            self.bm25encoder.fit(clean_document)
            self.bm25encoder.dump(self.bm25_store)
            self.bm25_corpus = True
            logging.info("Data have fit and dump")

        except Exception as e:
            logging.error(f"ERROR : {e}")
            raise e


        
    def bm25_DocumentEncoder(self, document: List[Document]):

        try:

            if not isinstance(document[0], Document):
                logging.errro("document data should be langchain Document")
                raise ValueError("document data should be langchain Document")
            
            clean_document = self.clean_data(document)

            self.bm25_Fit(document) 

            if self.bm25_corpus:
                encode_document = self.bm25encoder.encode_documents(clean_document)
                logging.info("convert text data into sparse vector")
                return encode_document

            logging.error(f"data have not fited. till bm25_fit is {self.bm25_corpus}")
            raise ValueError("data have not bm25 fit ")

        except Exception as e:
            print(f"ERROR : {e}")
            raise e


        

    def bm25_QueryEncoder(self, query: str):

        try:

            if self.bm25loaded:

                query = query.lower().strip()
                encode_query = self.bm25encoder.encode_queries(query)
                logging.info("user query have convert into sparse vector")
                return encode_query

            logging.error("document have not fit or fited not loaded")
            raise ValueError("document have not fit or fited not loaded")
        
        except Exception as e:
            logging.error(f"ERROR : {e}")
            raise e