#---------------------------------------------------------------------------------------------------
# Dependencies 
#---------------------------------------------------------------------------------------------------

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance 
from typing import Literal, Dict, List, Annotated, Optional

# local 
from pdf_processor import PDFProcessor
from agent import PodagentConfigs
from qdrant_manager import setup_temp_qdrant

# built in
from configs import QDRANT_CLIENT_URL, EMBEDDING_MODEL





# instance of pdf processor 
pdf_process = PDFProcessor()





#---------------------------------------------------------------------------------------------------
# Class scripts
#---------------------------------------------------------------------------------------------------

class ConversationalAgenticRAG:

    SUPPORTED_FILES = ["pdf"]



    def __init__(
            self,
            file_path: str 
        ):
        
        """ constructor """

        self.__file_path = file_path
        self.__chunk_size = 1000
        self.__overlap_size = 200
        self.__embedding_model = None
        self.__qdrant_collection_name = "conv_agentic_rag_db"

        self.__qdrant_client = QdrantClient(url=QDRANT_CLIENT_URL)
        self.__qdrant_vector_db: QdrantVectorStore = None 

        self.__pdf_texts: List[Document] = None

        # setup everything here
        self.__indexing()


        
    def __repr__(self):
        return "ConversationalAgenticRAG()"
    



    def __load_pdf_texts(self):
        """ It will load pdf content and update in the variable """
        self.__pdf_texts = pdf_process.process_pdf(pdf_path=PodagentConfigs.pdf_path)
        




    def __load_embedding_model(self):
        """ It will load embedding model """
        self.__embedding_model = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL
        )






    def __indexing(self):
        """ Indexing - A step in RAG """

        # loading embedding model
        self.__load_embedding_model()

        # loading content 
        self.__load_pdf_texts()

        # creating vector db
        self.__create_qdrant_vector_db()





    def __create_qdrant_vector_db(self):
        
        # self.__qdrant_client.recreate_collection(
        #     collection_name=self.__qdrant_collection_name,
        #     vectors_config=VectorParams(
        #         # supports: 128 - 3072, Recommended: 768, 1536, 3072 
        #         size=1024,
        #         distance=Distance.COSINE
        #     )
        # )

        # creating vector database
        self.__qdrant_vector_db = QdrantVectorStore(
            client=self.__qdrant_client,
            collection_name=self.__qdrant_collection_name,
            embedding=self.__embedding_model
        )

        return True





    def fetch_docs(
            self, 
            query: str,
            tok_k: int = 3,
            conversational: bool = True,
            last_five_chat_messages: Optional[List[str]] = None
        ) -> List[str]:

        pass 







if __name__ == "__main__":

    rag = ConversationalAgenticRAG(file_path=PodagentConfigs.pdf_path)



