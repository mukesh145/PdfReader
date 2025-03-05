import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain_openai import OpenAIEmbeddings, OpenAI

class HRPolicyIndexer:
    def __init__(self, pdf_url: str, openai_api_key: str):
        self.pdf_url = pdf_url
        self.openai_api_key = openai_api_key
        self.index = None

    def load_data(self):
        """Load data from the given PDF URL."""
        return PyPDFLoader(self.pdf_url)

    def split_text(self):
        """Split text using recursive character splitting."""
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=10
        )

    def create_embeddings(self):
        """Create text embeddings using OpenAI."""
        return OpenAIEmbeddings(openai_api_key=self.openai_api_key)

    def create_index(self):
        """Create and store the vector database index."""
        data_loader = self.load_data()
        text_splitter = self.split_text()
        embeddings = self.create_embeddings()
        
        index_creator = VectorstoreIndexCreator(
            text_splitter=text_splitter,
            embedding=embeddings,
            vectorstore_cls=FAISS
        )
        
        self.index = index_creator.from_loaders([data_loader])
        return self.index

class HRRAG:
    def __init__(self, index: HRPolicyIndexer, openai_api_key: str):
        self.index = index
        self.openai_api_key = openai_api_key
        self.llm = self.connect_llm()

    def connect_llm(self):
        """Connect to OpenAI GPT model."""
        return OpenAI(api_key=self.openai_api_key, model="gpt-4", temperature=0.1)

    def get_response(self, question: str):
        """Retrieve the best match from the Vector DB and send it to the LLM."""
        if self.index.index is None:
            raise ValueError("Index has not been created yet. Please create the index first.")
        
        return self.index.index.query(question=question, llm=self.llm)

# Usage Example
if __name__ == "__main__":
    pdf_url = 'https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf'
    openai_api_key = "your_openai_api_key_here"
    
    indexer = HRPolicyIndexer(pdf_url, openai_api_key)
    index = indexer.create_index()
    
    hr_rag = HRRAG(index, openai_api_key)
    response = hr_rag.get_response("What is the leave policy in India?")
    print(response)
