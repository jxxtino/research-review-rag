from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.embeddings import generate_embedding
from src.documents import DocumentsHandler

class ChromaDB:
    def __init__(self, CHROMA_DB_PATH: str = "database/"):
        self.chroma_db_path = CHROMA_DB_PATH
        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            max_length=512,
            device="cpu")


        self.db = Chroma(
            collection_name="RAG_PDF_DOCUMENTS",
            persist_directory=self.chroma_db_path,
            embedding_function=generate_embedding())

    def insert(self, chunks: list[Document]):
        existing_items = self.db.get(include=[])
        existing_ids = set(existing_items["ids"])

        chunks_ids = DocumentsHandler.calculate_chunk_id(chunks)
        new_chunks = []

        for chunk in chunks_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if new_chunks:
            new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
            self.db.add_documents(new_chunks, ids=new_chunks_ids)

    def search(self, query: str, k: int = 3):
        return self.db.similarity_search(query, k)

    def search_rerank(self, 
                      query: str, 
                      candidate_k: int = 30,
                      top_k: int = 8
                      ):
        
        candidates = self.search(query, candidate_k)
        pairs = [(query, doc.page_content) for doc in candidates]

        scores = self.reranker.predict(
            pairs,
            batch_size=4,
            show_progress_bar=False)

        ranked = sorted(
            zip(candidates, scores),
            key= lambda item: float(item[1]),
            reverse=True)

        results = []

        for doc, score in ranked[:top_k]:
            doc.metadata["rerank_ccore"] = float(score)
            results.append(doc)

        return results
    
    def clean_database(self):
        self.db.reset_collection()
            
        