import json
from pprint import pprint
import os
import time
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

elastic_user=os.getenv('ES_USER')
elastic_password=os.getenv('ES_PASSWORD')
elastic_endpoint=os.getenv("ES_ENDPOINT")

class Search:
    def __init__(self):
        url = f"https://{elastic_user}:{elastic_password}@{elastic_endpoint}:9200"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.es = Elasticsearch(url, ca_certs = "./http_ca.crt", verify_certs = True) 
        client_info = self.es.info()
        print('Connected to Elasticsearch!')
        pprint(client_info.body)
        
    def create_index(self):
        self.es.indices.delete(index='my_documents', ignore_unavailable=True)
        self.es.indices.create(index='my_documents', mappings={
            'properties': {
                'embedding': {
                    'type': 'dense_vector',
                }
            }
        })
        
    def get_embedding(self, text):
        return self.model.encode(text)
        
    def insert_document(self, document):
        return self.es.index(index='my_documents', document={
            **document,
            'embedding': self.get_embedding(document['summary']),
        })
    
    def insert_documents(self, documents):
        operations = []
        for document in documents:
            operations.append({'index': {'_index': 'my_documents'}})
            operations.append({
                **document,
                'embedding': self.get_embedding(document['summary']),
            })
        return self.es.bulk(operations=operations)
    
    def reindex(self):
        self.create_index()
        with open('data.json', 'rt') as f:
            documents = json.loads(f.read())
        return self.insert_documents(documents)

    def search(self, **query_args):
        return self.es.search(index='my_documents', **query_args)
    
    def retrieve_document(self, id):
        return self.es.get(index='my_documents', id=id)
