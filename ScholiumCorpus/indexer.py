import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.utilities.arxiv import ArxivAPIWrapper

from dotenv import load_dotenv

load_dotenv()


pinecone_api_key = os.environ.get("PINECONE_API_KEY")

def _lazy_load_pdf(file_path):
    loader = ArxivLoader(
                query= file_path,
                load_max_docs=10,
                load_all_available_meta=True)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)
    return pages

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # chunk size (characters)
    chunk_overlap=20,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)

def arXiv_search(query, max_docs=10):
    '''
    Loads the max_docs documents from arxiv.
    Args:
        query (str): The query to search for.
        max_docs (int): The maximum number of documents to return.

    Returns:
        list[Document]: A list of documents
    '''
    # For some stupid reason, the ArxivLoader doesn't work?
    client = ArxivAPIWrapper(
        top_k_results = max_docs,
        load_max_docs = max_docs,
        load_all_available_meta = True,
    )
    return client.load(query=query)
 


class Indexer():
    def __init__(self, index_name = "scholium-index"): 
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.pc = Pinecone(api_key= pinecone_api_key)

        # Check if index exists
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        if index_name not in existing_indexes: 
            raise ValueError(f"Index {index_name} does not exist")
        else:
            self.index = self.pc.Index(index_name)
        
        self.vector_store = PineconeVectorStore(embedding=self.embeddings, index=self.index)

    def index_documents(self, documents: list[Document]):   
        try:
            for doc in documents:
                splits = text_splitter.split_documents([doc])
                splits = self._replace_all_none_values(splits)
                self.vector_store.add_documents(documents=splits)
        except Exception as e:
            print(f"Error indexing document: {e}")
        
    def _replace_none_values(self,d):
        try:
            for k, v in d.items():
                if v is None:
                    d[k] = "None"
        except Exception as e:
            print(f"Error replacing none values: {e}")

    def _replace_all_none_values(self, splits):
        for split in splits:
            self._replace_none_values(split.metadata)
        return splits
    
    def index_query(self, query, max_docs=10):
        documents = arXiv_search(query, max_docs)
        self.index_documents(documents)


if __name__ == "__main__":
    vector_store = InMemoryVectorStore(OpenAIEmbeddings())
    num_docs = 100
    query = "Transformers"
    documents = arXiv_search(query, num_docs )
    assert len(documents) == num_docs
    indexer = Indexer("test-index")
    indexer._replace_all_none_values(documents)
    for doc in documents:
        for k, v in doc.metadata.items():
            if v is None:
                raise ValueError(f"Metadata value is None for key: {k}")
    print("Metadata is valid")
    indexer.index_query(query=query,max_docs=num_docs)

