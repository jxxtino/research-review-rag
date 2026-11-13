from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentsHandler:
    def __init__(self, DATA_FOLDER_PATH):
        self.data_path = DATA_FOLDER_PATH
        Path(self.data_path).mkdir(parents=True, exist_ok=True)

    def load_documents(self):
        loader = PyPDFDirectoryLoader(self.data_path)
        return loader.load()
    
    @staticmethod
    def calculate_chunk_id(chunks):
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            page_id = f"{source}:{page}"

            if page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            chunk_id = f"{page_id}:{current_chunk_index}"
            last_page_id = page_id

            chunk.metadata["id"] = chunk_id

        return chunks


    def split_documents(self, documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        return text_splitter.split_documents(documents)
