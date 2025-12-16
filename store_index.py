from dotenv import load_dotenv
import os

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    load_embeddings
)

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# 1️⃣ Load and prepare documents
print("Loading PDF documents...")
extracted_data = load_pdf_file(data_dir="data/")
filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered_data)

print(f"Total chunks created: {len(text_chunks)}")


# 2️⃣ Load embeddings (OpenAI – Arabic & French friendly)
embeddings = load_embeddings()


# 3️⃣ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "raqaba-ai"   # lowercase only

# 4️⃣ Create index if not exists
if not pc.has_index(index_name):
    print("Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=1536,              # OpenAI embedding dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

index = pc.Index(index_name)

print("Index stats BEFORE upsert:")
print(index.describe_index_stats())


# 5️⃣ Store documents in Pinecone
print("Upserting documents into Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("Index stats AFTER upsert:")
print(index.describe_index_stats())

print("✅ Indexing completed successfully.")
