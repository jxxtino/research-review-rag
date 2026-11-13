from src.documents import DocumentsHandler
from src.database import ChromaDB
from src.model import LLM

database = ChromaDB()
model = LLM()

doc_handler = DocumentsHandler(DATA_FOLDER_PATH="docs/")
documents = doc_handler.load_documents()

chunks = doc_handler.split_documents(documents)
database.insert(chunks)

question = "quais métodos os artigos usam, e como eles se comparam?"

results = database.search_rerank(query=question, 
                                 candidate_k=15,
                                 top_k=5)

context = "\n\n".join(
    f"Artigo: {doc.metadata.get("title", "Desconhecido")}\n"
    f"Página: {doc.metadata.get("page", 0) + 1}\n"
    f"Conteúdo: {doc.page_content}"
    for doc in results)

response = model.response(
    query = question,
    context=context)

print(response)
