from ScholiumCorpus.indexer import Indexer

queries = ["Attention Based Transformers", "Encoder-Decoder Models",
            "Bidirectional Encoders", "DeepSeek", "Chain of Though Prompting",
            "Positional Encoding", "Positional Embedding", "LLaMA", "GPT","Large Language Models"]

max_docs = 100
index_name = "test-index"

indexer = Indexer(index_name)
for query in queries:
    indexer.index_query(query, max_docs)
    print("Done indexing: {query}")
print("Done indexing")