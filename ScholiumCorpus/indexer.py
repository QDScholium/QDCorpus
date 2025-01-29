import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.vectorstores import InMemoryVectorStore

from dotenv import load_dotenv

load_dotenv()


COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
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
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)


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

    def index_document(self, file_path):
        pages = _lazy_load_pdf(file_path)
        all_splits = text_splitter.split_documents(pages)
        self.vector_store.add_documents(documents=all_splits)

    def replace_none_values(d):
        return {k: ('None' if v is None else v) for k, v in d.items()}

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "CoT.pdf")
    indexer = Indexer(index_name='test-index')
    # pages = _lazy_load_pdf(file_path)
    # print(pages[2])
    # splits = text_splitter.split_documents(pages)
    # print(splits[2])
    # print(type(splits[2]))
    # indexer.index_document(file_path)
    # arXiv = _lazy_load_pdf("Chain of Thought")
    arXivloader = ArxivLoader(
                query= file_path,
                load_max_docs=10,
                load_all_available_meta=True)
    # for page in pages:
    #     print(page.metadata)
    vector_store = InMemoryVectorStore(OpenAIEmbeddings)
    pages = []
    for page in arXivloader.lazy_load():
        pages.append(page)
    
    for p in pages:
        all_splits = text_splitter.split_documents(pages)
        vector_store.add_documents(all_splits)
    
    # for pages in arXiv:
    #     all_splits = text_splitter.split_documents(pages)
    #     vector_store.add_documents(all_splits)
    # print("Done!")


    # all_splits = text_splitter.split_documents(pages)
    # print(all_splits[5].metadata)
    # for split in all_splits:
    #     print(type(split.metadata))


