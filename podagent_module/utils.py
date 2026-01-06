
# ----------------------------------------------------------------------------------------------------------------------------
#  Dependencies
# ----------------------------------------------------------------------------------------------------------------------------

from langchain_core.documents import Document
from typing import List 
import fitz



def get_clean_chunks(chunks: List[Document], plain_text: bool = False) -> str:
    
    result = ""

    for chunk in chunks:
        if not plain_text:
            result += f"\n\n{10*'-'}\n"
            result += f"[Chunk page number: {chunk.metadata['page']}]\n"
            
        result += chunk.page_content 

    result += f"\n\n{10*'-'}\n\n"
    return result





def load_pdf_content(pdf_path: str, _from: int, to: int) -> List[Document]:
    """load content of pdf

    Args:
        pdf_path (str): memory path
        _from (int, optional): starting page. Defaults to 0.
        to (int, optional): page ending. Defaults to 0.

    Returns:
        List[Document]: list of pages content
    """
    doc = fitz.open(pdf_path)
    documents = []

    for page_no in range(_from,to):
        page = doc.load_page(page_no)
        text = page.get_text()
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "page": page_no,
                    "source": pdf_path
                }
            )
        )
    return documents







def save_workflow_diagram(workflow, name: str = "workflow.png"):

    # ------------------------- saving workflow diagram ----------------
    png_bytes = workflow.get_graph().draw_mermaid_png()

    with open(name, "wb") as f:
        f.write(png_bytes)






