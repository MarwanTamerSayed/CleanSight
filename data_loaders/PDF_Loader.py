from pathlib import Path
import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def pdf_loader(folder_path):

    folder_path = Path(folder_path)

    documents = []

    # STEP 1 → Create page-level documents
    for file in folder_path.glob("*.pdf"):

        with fitz.open(file) as doc:

            for page_num, page in enumerate(doc):

                text = page.get_text()

                if text.strip():

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(file),
                                "page": page_num + 1,
                                "type": "pdf"
                            }
                        )
                    )

    # STEP 2 → Recursive chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    return chunks