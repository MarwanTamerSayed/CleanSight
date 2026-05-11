from pathlib import Path
import pandas as pd
from langchain_core.documents import Document
import fitz

def spread_sheet_loader(folder_path):
    folder_path = Path(folder_path)
    documents = []
    
    for pattern in ('*.xls', '*.xlsx'):
        for file in folder_path.glob(pattern):
            df = pd.read_excel(file)
            for _, row in df.iterrows():
                text = "\n".join(
                    [f"{col}: {value}" for col, value in row.items()]
                )
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(file)} 
                    )
                )
    return documents


