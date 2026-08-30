import uuid
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter





class TextChunking:

    def recursiveCharacterTextSplitter(self,
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

            NAMESPACE = uuid.NAMESPACE_DNS
            final_chunk : List[Document] = []

            for i , chunk in enumerate(recursive_text_chunking , start = 1):
                u_id = uuid.uuid5(NAMESPACE , chunk.page_content)
                chunk.metadata.setdefault("chunk_id" , str(u_id))
                chunk.metadata.setdefault("chunk_index" , i)

                final_chunk.append(chunk)
            return final_chunk
            
        except Exception as e:
            print(f"ERROR : {e}")
            raise e