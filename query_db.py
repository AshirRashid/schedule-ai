import os
import json
import chromadb
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb import EmbeddingFunction, Documents, Embeddings


class CustomHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def __call__(self, input: Documents) -> Embeddings:
        return np.array(self.embed_documents(input))


text_to_match = "get all possible chunks that look like they are referring to a possible event"
embedding_function = CustomHuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
# embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

chroma_client = chromadb.HttpClient(host='localhost', port=8000)
collection = chroma_client.get_collection(
    name="langchain", embedding_function=embedding_function)

results = collection.query(
    query_texts=[text_to_match],
    n_results=1,
    include=["documents", "metadatas", "distances"]
)

print(results["documents"][0][0])
breakpoint()
