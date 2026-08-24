from langchain_core.documents import Document

from Chroma.Chroma import create_vector_store, get_retriever


# Test documents
documents = [
    Document(
        page_content="Revenue increased by 25% during the financial year.",
        metadata={"source": "test1"}
    ),

    Document(
        page_content="The company reported a net profit of $5 million.",
        metadata={"source": "test2"}
    ),

    Document(
        page_content="The company has 150 employees across different departments.",
        metadata={"source": "test3"}
    ),

    Document(
        page_content="Operating expenses decreased by 10% compared to last year.",
        metadata={"source": "test4"}
    ),
]


# Create Chroma database
vector_store = create_vector_store(
    documents,
    db_name="test_db"
)


print("\nChroma database created successfully!")


# Create retriever
retriever = get_retriever(
    vector_store,
    k=2
)


# Test query
query = "How much did the company's revenue increase?"

results = retriever.invoke(query)


print("\nQUERY:")
print(query)

print("\nRESULTS:")

for i, doc in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)