Basic Offline RAG (Retrieval-Augmented Generation)

Flow:
Corpus (Data) -> Chunks -> Embed -> Vector DB -> LLM (User query + prompt) -> Response

v2 -> Basic chunking, FAISS (In-memory docstore) as Vector DB, LLama 3.2 as Embedding Model & LLM
v3 -> Dynamic chunking with context, Weaviate DB as Vector DB, LLama 3.2 as Embedding Model & LLM

Scope for Improvement:
-> Build a UI
-> Cache the VDB offline for computing constraints
-> Mix and match different chunking algo, VDB & LLMs