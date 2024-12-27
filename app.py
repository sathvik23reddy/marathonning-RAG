import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Step 2: Chunk the Text
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Step 3: Embed the Chunks
def embed_chunks(chunks, embedder):
    embeddings = embedder.encode(chunks, convert_to_tensor=True)
    return embeddings.cpu().numpy()  # Ensure embeddings are on the CPU and in NumPy format

# Step 4: Set Up a Vector Store
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Step 5: Retrieve Relevant Chunks
def get_relevant_chunks(question, embedder, index, chunks, k=5):
    question_embedding = embedder.encode(question, convert_to_tensor=True).cpu().numpy()
    distances, indices = index.search(np.array([question_embedding]), k)
    return [chunks[i] for i in indices[0]]

# Step 6: Generate Answers
def generate_answer(generator, tokenizer, context, question):
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    outputs = generator.generate(**inputs, max_new_tokens=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main Interactive Function
def main():
    # Load PDF and preprocess
    print("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf('adv-mara.pdf')
    print("Chunking text...")
    chunks = chunk_text(pdf_text)

    # Embed the chunks
    print("Embedding text chunks...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')  # Use CPU explicitly
    chunk_embeddings = embed_chunks(chunks, embedder)

    # Create FAISS index
    print("Creating vector store...")
    index = create_faiss_index(chunk_embeddings)

    # Load Generator
    print("Loading generator model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    generator = GPT2LMHeadModel.from_pretrained("gpt2")

    print("\nSystem is ready. Type your question below (type 'exit' to quit):")
    while True:
        question = input(">>> ").strip()
        if question.lower() == 'exit':
            print("Exiting. Goodbye!")
            break

        # Retrieve and generate answer
        print("Retrieving relevant information...")
        relevant_chunks = get_relevant_chunks(question, embedder, index, chunks)
        context = " ".join(relevant_chunks)
        print("Generating response...")
        answer = generate_answer(generator, tokenizer, context, question)
        print(f"Answer: {answer}\n")

# Run the application
if __name__ == "__main__":
    main()
