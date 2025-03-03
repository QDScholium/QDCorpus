from ScholiumCorpus.indexer import Indexer

max_docs = 100
index_name = "test-index"
query = "Transformers"

indexer = Indexer(index_name)
indexer.index_query(query, max_docs)
print("Done indexing: {query}")
print("Done indexing")