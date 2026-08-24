from langchain_huggingface import HuggingFaceEmbeddings

def create_embedding_model():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        encode_kwargs={
            "normalize_embeddings": True
        },
        query_encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings