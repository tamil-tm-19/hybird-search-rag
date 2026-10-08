import uuid
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
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




class TextChunking:

    def recursive_CharacterTextSplitter(self,
                    document_data: list[Document],
                    chunks_size: int,
                    chunks_over_lap: int,
                    separators: List[str] | None = None)-> List[Document]:


        try:
            if separators is None:
                separators = ["\n\n", "\n", ". ", " ",""]

            recursive_chunking_setup = RecursiveCharacterTextSplitter(
                                                    separators = separators,
                                                    chunk_size = chunks_size,
                                                    chunk_overlap = chunks_over_lap)
            
            recursive_text_chunking = recursive_chunking_setup.split_documents(documents = document_data)
            logging.info(f"successfully split the text intp chunk {len(recursive_text_chunking)}")

            NAMESPACE = uuid.NAMESPACE_DNS
            final_chunk : List[Document] = []

            for i , chunk in enumerate(recursive_text_chunking , start = 1):
                u_id = uuid.uuid5(NAMESPACE , chunk.page_content)
                chunk.metadata.setdefault("chunk_id" , str(u_id))
                chunk.metadata.setdefault("chunk_index" , i)

                final_chunk.append(chunk)

            logging.info("update the chunk id & chunk index")
            return final_chunk
            
        except Exception as e:
            logging.error(f"{e}")
            raise e