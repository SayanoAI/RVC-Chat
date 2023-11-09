from functools import lru_cache
import hashlib
import json
import chromadb
import numpy as np
from uuid import uuid4
from webui.functions import load_functions

from webui.utils import gc_collect

@lru_cache
def get_db_client():
    client = chromadb.Client()
    return client

def get_collection_and_key(name,embedding_function=None):
    client = get_db_client()
    num = len(client.list_collections())

    # Create a collection for function calls
    key = hashlib.md5(f"{name}-{num}".encode('utf-8')).hexdigest()
    if embedding_function is None:
        collection = client.get_or_create_collection(key)
    else:
        collection = client.get_or_create_collection(key,embedding_function=embedding_function)

    return collection, key

class VectorDB:
    def __init__(self,name="",embedding_function=None):
        self.name=name
        self.embedding_function=embedding_function
        self.collection, self.key = get_collection_and_key(name,embedding_function=embedding_function)
        load_functions(self)

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
        finally:
            self.collection, self.key = get_collection_and_key(self.name)
            load_functions(self)

    def add_function(self,description,function,arguments,**args):
        self.collection.add(ids=[str(uuid4())],documents=description,metadatas={
            "hnsw:space": "cosine",
            "type": "function",
            "function": function,
            "arguments": json.dumps(arguments),
            "template": json.dumps(args)
            })

    def add_document(self,document,**kwargs):
        self.collection.add(ids=[str(uuid4())],documents=document,metadatas={
            "hnsw:space": "cosine",
            "type": "document",
            **kwargs
            })

    def get_query(self, query="",n_results=1,threshold=1.,include=[],type="document",verbose=False):
        # Query the collection using natural language
        query_results = self.collection.query(
            query_texts=query,
            n_results=n_results,
            include=include+["metadatas", "distances"],
            where={"type": type}
        )
        if verbose: print(f"{query_results=}")
        distances = np.array(query_results["distances"])
        index = np.where(distances<threshold)[0]
        results = []
        for i in index:
            results.append({
                **{field[:-1]:query_results[field][i][0] for field in include},
            })

        return results