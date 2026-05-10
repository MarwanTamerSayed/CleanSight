from pathlib import Path
import fitz
from langchain_core.documents import Document
import pandas as pd


def csv_loader(folder_path):
    folder_path = Path(folder_path)
    documents = []
    for file in folder_path.glob('*.csv'):
        df = pd.read_csv(file)

        for _, row in df.iterrows():
            text = "\n".join(
                [f"{col}: {value}" for col, value in row.items()]
            )

            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file}
                )
            )

    return documents        
            