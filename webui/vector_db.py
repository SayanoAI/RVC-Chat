from functools import lru_cache
import hashlib
import chromadb
import numpy as np
from uuid import uuid4

from webui.utils import gc_collect

@lru_cache
def get_db_client():
    client = chromadb.Client()
    return client

def get_collection(name):
    client = get_db_client()
    num = len(client.list_collections())

    # Create a collection for function calls
    key = hashlib.md5(f"{name}-{num}".encode('utf-8')).hexdigest()
    collection = client.get_or_create_collection(key)

    return collection, key

class VectorDB:
    def __init__(self,name=""):
        self.name=name
        self.collection, self.key = get_collection(name)

    def __del__(self):
        try:
            get_db_client().delete_collection(self.key)
            del self.collection
        except Exception as e: print(f"Failed to delete collection{e}")
        finally: gc_collect()

    def clear(self):
        try:
            get_db_client().delete_collection(self.key)
            del self.collection
        except Exception as e: print(f"Failed to delete collection{e}")
        finally: self.collection, self.key = get_collection(self.name)

    def add_documents(self,document,**kwargs):
        self.collection.add(ids=[str(uuid4())],documents=document,metadatas={
            "hnsw:space": "cosine",
            **kwargs
            })

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