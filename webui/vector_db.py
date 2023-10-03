from functools import lru_cache
import hashlib
import chromadb
import numpy as np

@lru_cache
def get_db_client():
    client = chromadb.Client()
    return client

@lru_cache
def get_collection(name):
    client = get_db_client()
    num = len(client.list_collections())

    # Create a collection for function calls
    key = hashlib.md5(f"{name}-{num}".encode('utf-8')).hexdigest()
    collection = client.create_collection(key)

    return collection

class VectorDB:
    def __init__(self,name=""):
        self.collection = get_collection(name)

    def add_documents(self,document,metadata=None):
        self.collection.add(documents=document,metadatas=metadata)

    def get_query(self, query="",n_results=1,threshold=1.,include=["metadatas", "distances"]):
        # Query the collection using natural language
        query_results = self.collection.query(
            query_texts=query,
            n_results=n_results,
            include=include
        )
        distances = np.array(query_results["distances"])
        index = np.where(distances<threshold)[0]
        results = []
        for i in index:
            results.append({
                **{field[:-1]:query_results[field][i][0] for field in include},
            })

        return results