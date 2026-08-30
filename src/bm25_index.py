import os
from typing import List, Dict, Any, Union
from pinecone_text.sparse import BM25Encoder
from langchain_core.documents import Document


class BM25encoder:

    def __init__(self, bm25_data_store: str = "meta_corpus.json"):

        try:
            self.bm25encoder = BM25Encoder()
            self.bm25_store = bm25_data_store
            self.bm25_corpus = False

            if os.path.exists(bm25_data_store):
                self.bm25encoder.load(path = bm25_data_store)
                self.bm25loaded = True
            else:
                self.bm25loaded = False

        except Exception as e:
            print(f"ERROR : {e}")
            raise e
        

    @staticmethod
    def clean_data(document: List[Document])-> List[str]:

        try:
            if isinstance(document[0], Document):
                text_cleaning = [doc.page_content.lower().strip() for doc in document]
                return text_cleaning

        except Exception as e:
            print(f"ERROR : {e}")
            raise e

        

    def bm25_Fit(self, document: List[Document]):

        try:
            if not isinstance(document[0], Document):
                raise ValueError("document data should be Document")
            
            clean_document = self.clean_data(document)
            self.bm25encoder.fit(clean_document)
            self.bm25encoder.dump(self.bm25_store)
            self.bm25_corpus = True

        except Exception as e:
            print(f"ERROR : {e}")
            raise e


        
    def bm25_DocumentEncoder(self, document: List[Document]):

        try:

            if not isinstance(document[0], Document):
                raise ValueError("document data should be Document")
            
            clean_document = self.clean_data(document)

            self.bm25_Fit(document) 

            if self.bm25_corpus:
                encode_document = self.bm25encoder.encode_documents(clean_document)
                return encode_document
            raise ValueError("bm25 fit ")

        except Exception as e:
            print(f"ERROR : {e}")
            raise e


        

    def bm25_QueryEncoder(self, query: str):

        try:

            if self.bm25loaded:

                query = query.lower().strip()
                encode_query = self.bm25encoder.encode_queries(query)
                return encode_query

            raise ValueError("document have not fit or fited not loaded")
        except Exception as e:
            print(f"ERROR : {e}")
            raise e