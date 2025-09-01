from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os, hashlib

load_dotenv()

# load pdfs -> split to chunks -> save to db

def main():
    docs = load_pdfs()
    chunks = split_text(docs)
    save_to_db(chunks) # postreSQL + pgvector

# load pdfs
def load_pdfs() -> list[Document]:
    docs = []
    base_path = Path(__file__).parent / "data"
    docs.extend(load_pdfs_from_dir(str(base_path / "AI"), "ai"))
    docs.extend(load_pdfs_from_dir(str(base_path / "Full stack"), "fullstack"))
    docs.extend(load_pdfs_from_dir(str(base_path / "Software"), "software"))
    return docs

def load_pdfs_from_dir(path, role):
    # directory does not exist
    if not os.path.exists(path):
        print(f"{path} does not exist")
        return []
    
    # list for storing every page on every pdf
    docs = []
    loaded = 0
    for file in sorted(os.listdir(path)): # pdfs
        if not file.endswith(".pdf"):
            continue
        try:
            pdf_path = Path(path) / file
            doc_id = make_doc_id(str(pdf_path))
            new_docs = PyPDFLoader(os.path.join(path, file)).load() # every page on a pdf
            for d in new_docs:
                d.metadata.update({
                    "role": role,
                    "file_path": str(pdf_path.resolve()),
                    "doc_id": doc_id,
                })
            loaded += 1
            docs.extend(new_docs)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    print(f"loaded {loaded} pdfs from {path}")
    return docs

# return hash from file path
def make_doc_id(file_path: str) -> str:
    return hashlib.sha1(Path(file_path).resolve().as_posix().encode()).hexdigest()

# split to chunks
def split_text(documents: list[Document]):
    # split every page into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", "•", ":", ".", " "]
    )
    chunks = []
    # group by doc_id
    docs_by_id = defaultdict(list)
    for d in documents:
        docs_by_id[d.metadata["doc_id"]].append(d)

    for doc_id, docs in docs_by_id.items():
        chunk_idx = 0
        for d in docs:
            split = splitter.split_documents([d]) # split every page into chunks
            for c in split:
                c.metadata.update({
                    "chunk_index": chunk_idx,
                    "page": d.metadata.get("page", 0)  # Extract page number
                })
                chunk_idx += 1
                chunks.append(c)
    return chunks

# save to db
def save_to_db(chunks):
    # Create embeddings object
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    CONNECTION_STRING = os.getenv("CONNECTION_STRING")

    # Create PGVector store
    vectorstore = PGVector(
        collection_name="job_chunks",
        connection=CONNECTION_STRING,
        embeddings=embedding_function
    )
    
    # Add documents
    vectorstore.add_documents(chunks)
    print(f"Saved {len(chunks)} chunks to PGVector database")

if __name__ == "__main__":
    main()

"""
# add
ids = vectorstore.add_texts(
    texts=[chunk_text_1, chunk_text_2],
    metadatas=[{"doc_id": "...", "role":"ai", "source_path":"...", "page": 5}, ...]
)
"""
