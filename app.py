from langchain_ollama import OllamaEmbeddings
from PyPDF2 import PdfReader
from weaviate.classes.init import AdditionalConfig, Timeout
import spacy
import weaviate
import requests
import json
import weaviate
from tqdm import tqdm

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Step 2: Chunk the Text Dynamically
def chunk_text_dynamic(text, min_chunk_size=200, max_chunk_size=500, overlap_factor=0.3):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    chunks = []
    temp_chunk = ""
    for sent in tqdm(doc.sents, desc="Chunking Text", unit="sentences"):
        if len(temp_chunk) + len(sent.text) < max_chunk_size:
            temp_chunk += " " + sent.text.strip()
        else:
            chunks.append(temp_chunk.strip())
            temp_chunk = sent.text.strip()

    # Add the final chunk
    if temp_chunk:
        chunks.append(temp_chunk.strip())

    # Add overlap
    overlapping_chunks = []
    overlap = int(max_chunk_size * overlap_factor)
    for i in range(len(chunks)):
        overlapping_chunks.append(chunks[i])
        if i > 0 and len(chunks[i]) > min_chunk_size:
            overlapping_chunks[-1] = chunks[i - 1][-overlap:] + " " + chunks[i]

    return overlapping_chunks

# Step 3: Set Up Weaviate Client
def setup_weaviate():
    client = weaviate.connect_to_local(
    port=8080,
    grpc_port=50051,
    additional_config=AdditionalConfig(
        timeout=Timeout(init=30, query=60, insert=120)  
        )
    )

    try:
        questions = client.collections.create(
            name="Question",
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),    
            generative_config=wvc.config.Configure.Generative.cohere(),             
            properties=[
                wvc.config.Property(
                    name="question",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="answer",
                    data_type=wvc.config.DataType.TEXT,
                ),
                wvc.config.Property(
                    name="category",
                    data_type=wvc.config.DataType.TEXT,
                )
            ]
        )
        print(questions.config.get(simple=False))
    finally:
        return client


# Step 4: Embed and Store Chunks in Weaviate
def embed_chunks_weaviate(chunks):
    client = setup_weaviate()
    embeddings = OllamaEmbeddings(model="llama3.2")

    for i, chunk in tqdm(enumerate(chunks), desc="Embedding Chunks", total=len(chunks), unit="chunk"):
        embedding = embeddings.embed_query(chunk)
        document = {
            "content": chunk,
            "metadata": f"Chunk-{i}"
        }
        client.collections.get("Document").data.insert(
            properties=document,
            vector=embedding
        )

    return client

# Step 5: Retrieve Relevant Chunks
def retrieve_relevant_chunks(question, client):
    embeddings = OllamaEmbeddings(model="llama3.2")
    query_vector = embeddings.embed_query(question)

    results = client.collections.get(
        "Document"
    ).query.near_vector(
        near_vector=query_vector, 
        limit = 5
    )

    documents = []

    for result in results.objects:
        documents.append(result.properties.get('content', None))

    return documents


# Step 6: Generate Answers
def generate_answer(user_input, relevant_document):
    full_response = []
    prompt = """Use the following pieces of context to answer the question at the end. If you don't know the answer, 
    just say that you don't know and do not make up an answer. Provide the answer in as much detail as possible, 
    covering all relevant aspects. Structure your response clearly and thoroughly.
    {relevant_document}

    Question: {user_input}

    Detailed Answer:"""
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
    pdf_file = input("Enter name of pdf to extract: ")
    
    # Load PDF and preprocess
    print("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(pdf_file)

    print("Chunking text dynamically...")
    chunks = chunk_text_dynamic(pdf_text)

    # Embed the chunks & create Weaviate Index
    print("Embedding text chunks into Weaviate...") 
    client = embed_chunks_weaviate(chunks)

    print("\nSystem is ready. Type your question below (type 'exit' to quit):")
    while True:
        question = input(">>> ").strip()
        if question.lower() == 'exit':
            print("Exiting. Goodbye!")
            break

        # Retrieving relevant chunks
        print("Retrieving relevant chunks...") 
        retrieved_docs = retrieve_relevant_chunks(question, client)

        # Combine retrieved docs into a single context string
        context = " ".join(retrieved_docs)

        # Generate response
        print("Generating response from llama3.2...") 
        print("Response>>>", end="")
        generate_answer(question, context)
    
    client.close()

# Run the application
if __name__ == "__main__":
    main()
