import os
from pathlib import Path
from typing import List, Union
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document





class DocumentLoader:

    def pdf_Parse(self, file_or_dir_path: str | Path)-> List[Document]:

        try:

            parse_data_lst = []

            if not file_or_dir_path[-4:] == ".pdf":

                path = Path(file_or_dir_path)
                if os.path.exists(path):
                    glob = list(path.rglob("**.pdf"))

                    for pdfs in glob:
                        pdf_loader = PyMuPDFLoader(str(pdfs))
                        pdf_loaded = pdf_loader.lazy_load()
                        print(f"{pdfs.name} is loaded successfully\n")

                        for doc in pdf_loaded:
                            doc.setdefault("source",pdfs.name)
                            doc.setdefault("file_path",path.name)
                            parse_data_lst.append(doc)
                        print(f"{pdfs.name} successfully Done\n")

                    return parse_data_lst

                raise FileNotFoundError(f"{path} File have not Found")
            
            pdf_loader = PyMuPDFLoader(str(file_or_dir_path))
            pdf_loaded = pdf_loader.lazy_load()

            for doc in pdf_loaded:
                doc.setdefault("file_path",file_or_dir_path)
                parse_data_lst.append(doc)

            return parse_data_lst

        except Exception as e:
            print(f"{e}")
            raise e