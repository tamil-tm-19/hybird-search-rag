import os
from pathlib import Path
from typing import List, Union
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
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



class DocumentLoader:

    def __init__(self, file_or_dir_path: str):

        self.file_path = file_or_dir_path


    def pdf_parse(self):
    
            if not os.path.exists(self.file_path):
                logging.error(f"PDF file not found: {self.file_path}")
                raise FileNotFoundError(f"PDF file not found: {self.file_path}")
    
            try:
    
                logging.info("file on process........ ")
                path = Path(self.file_path)
                glob = list(path.rglob("*.pdf"))
    
    
                logging.info("extracting the text from pdf")
                for i in glob:
                    load_pdf_data = PyMuPDFLoader(i)
                    pdf_data_loaded = load_pdf_data.load()
    
    
                return pdf_data_loaded
    
    
            except Exception as e:
                logging.error(f"{e}")
                raise e
    
    
    def pdf_parse(self):
    
            if not os.path.isfile(self.file_path):
                logging.error(f"PDF file not found: {self.file_path}")
                raise FileNotFoundError(f"PDF file not found: {self.file_path}")
    
            try:
                load_pdf_data = PyMuPDFLoader(self.file_path)
                pdf_data_loaded = load_pdf_data.load()
                logging.info("pdf text data have parsed.")
    
                return pdf_data_loaded
    
    
            except Exception as e:
                logging.error(f"{e}")
                raise e