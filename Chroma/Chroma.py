import os
from langchain_community.vectorstores import Chroma

BASE_DB_DIR = os.path.abspath("../chroma_dbs")

def create_vector_store(chunks, db_name="default_db"):
    db_path = os.path.join(BASE_DB_DIR, db_name)

    # Create folder if it doesn't exist
    os.makedirs(db_path, exist_ok=True)

    # Check if Chroma DB already exists
    if os.listdir(db_path):
        print(f"Loading existing DB: {db_name}")

        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )

        # Add new documents
        vector_store.add_documents(chunks)

    else:
        print(f"Creating new DB: {db_name}")

        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=db_path
        )

    vector_store.persist()

    return vector_store




def get_retriever(vector_store, k=3, fetch_k=10, lambda_mult=0.5):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult
        }
    )
    return retriever