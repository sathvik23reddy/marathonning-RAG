from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from PyPDF2 import PdfReader
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import requests
import json

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Step 2: Chunk the Text
def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
    all_splits = text_splitter.split_text(text)
    return text_splitter.create_documents(all_splits)

# Step 3: Embed chunks & set up vector store
def embed_chunks(chunks):
    embeddings = OllamaEmbeddings(model="llama3.2")
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    _ = vector_store.add_documents(documents=chunks)
    return vector_store

# Step 4: Generate Answers
def generate_answer(user_input, relevant_document):
    full_response = []
    prompt = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.

    {relevant_document}

    Question: {user_input}

    Helpful Answer:"""
    url = 'http://localhost:11434/api/generate'
    data = {
        "model": "llama3.2",
        "prompt": prompt.format(user_input=user_input, relevant_document=relevant_document)
    }
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, data=json.dumps(data), headers=headers, stream=True)
    try:
        for line in response.iter_lines():
            # filter out keep-alive new lines
            if line:
                decoded_line = json.loads(line.decode('utf-8'))
                full_response.append(decoded_line['response'])
    finally:
        response.close()
    print(''.join(full_response))

# Main Interactive Function
def main():
    # Load PDF and preprocess
    print("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf('your-pdf-here.pdf')
    print("Chunking text...")
    chunks = chunk_text(pdf_text)

    # Embed the chunks & create FAISS Index
    print("Embedding text chunks...") 
    vector_store = embed_chunks(chunks)

    print("\nSystem is ready. Type your question below (type 'exit' to quit):")
    while True:
        question = input(">>> ").strip()
        if question.lower() == 'exit':
            print("Exiting. Goodbye!")
            break

        #Retrieving relevant chunks
        print("Retrieving relevant chunks...") 
        retrieved_docs = vector_store.similarity_search(question)
        context = {"context": retrieved_docs}
        
        #Generate response
        print("Generating response from llama3.2...") 
        print("Response>>>", end="")
        generate_answer(question, context)

# Run the application
if __name__ == "__main__":
    main()
