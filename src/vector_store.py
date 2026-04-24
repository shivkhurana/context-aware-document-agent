import os
import pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.docstore.document import Document


def initialize_pinecone():
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),
        environment=os.getenv("PINECONE_ENV"),
    )
    index_name = "rag-index"
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=384, metric="cosine")
    return Pinecone.from_existing_index(index_name, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))


def embed_and_store_document(filename: str, content: str):
    """Embed and store a document in Pinecone vector store."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(content)

    docs = [Document(page_content=text, metadata={"source": filename}) for text in texts]

    vectorstore = initialize_pinecone()
    vectorstore.add_documents(docs)
    print(f"Embedded and stored document: {filename}")


# Global vectorstore instance
vectorstore = initialize_pinecone()
