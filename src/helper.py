from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain.schema import Document


# 1️⃣ Load PDF files
def load_pdf_file(data_dir: str) -> List[Document]:
    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# 2️⃣ Keep minimal metadata (source + page)
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []

    for doc in docs:
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": doc.metadata.get("source"),
                    "page": doc.metadata.get("page")
                }
            )
        )
    return minimal_docs


# 3️⃣ Split text – optimized for legal documents
def text_split(extracted_data: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # أكبر باش ما يتقطعش السياق القانوني
        chunk_overlap=200,      # overlap محترم
        separators=[
            "\nالمادة",
            "\nالفصل",
            "\nالباب",
            "\n",
            " "
        ]
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# 4️⃣ OpenAI embeddings (أفضل للعربية + الفرنسية)
def load_embeddings():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    return embeddings
