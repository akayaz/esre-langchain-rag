import json
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import ElasticsearchStore
from elasticsearch import Elasticsearch

load_dotenv()

# Elastic Cloud
CLOUD_ID = os.getenv('CLOUD_ID')
CLOUD_USERNAME = os.getenv('CLOUD_USERNAME')
CLOUD_PASSWORD = os.getenv('CLOUD_PASSWORD')
ES_VECTOR_INDEX = os.getenv('HOSTED_VECTOR_INDEX')
PIPELINE_ID = os.getenv('PIPELINE_ID')
DATASET = os.getenv('DATASET_PATH')+os.getenv('SERVICE_PUBLIC_FR_DATASET')

# Target Index name
index_name = "search-service-public-minilm-l6-v2"

# Create an Elasticsearch vector store with hosted model
def init_es_vectorstore_with_hosted_model():
    es_vector_store = ElasticsearchStore(
        es_cloud_id=CLOUD_ID,
        es_user=CLOUD_USERNAME,
        es_password=CLOUD_PASSWORD,
        index_name=ES_VECTOR_INDEX,
        query_field='text_field',
        vector_query_field='vector_query_field.predicted_value',
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(query_model_id="sentence-transformers__all-minilm-l6-v2")
    )
    return es_vector_store


INDEX_SETTINGS = {"index": { "default_pipeline": PIPELINE_ID}}

# create index with vector field
def create_index_with_vector_field(es_index_name, es_index_mapping):
    es = Elasticsearch(
        cloud_id=CLOUD_ID,
        basic_auth=(CLOUD_USERNAME, CLOUD_PASSWORD)
    )
    if es.indices.exists(index=es_index_name):
        es.indices.delete(index=es_index_name, ignore=[400, 404])

    es.indices.create(index=es_index_name,
                      mappings=es_index_mapping,
                      settings=INDEX_SETTINGS,
                      ignore=[400, 401])

# split dataset into passages
def split_docs_into_passages(dataset):
    metadata = []
    content = []

    # load dataset
    dataset_docs = json.loads(dataset.read())
    for docs in dataset_docs:
        content.append(docs["content"])
        metadata.append({
            "url": docs["url"],
            "title": docs["title"]
        })
    #text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, separators = ['\n'])
    return text_splitter.create_documents(content, metadatas=metadata)

# index data into elasticsearch
# we will set query_model_id to 'minilm-l6-v2' to embed dense vectors
def index_passages_into_es(dataset):
    docs_to_embed = split_docs_into_passages(dataset)
    documents = ElasticsearchStore.from_documents(
        docs_to_embed,
        es_cloud_id=CLOUD_ID,
        es_user=CLOUD_USERNAME,
        es_password=CLOUD_PASSWORD,
        index_name=ES_VECTOR_INDEX,
        query_field="text_field",
        vector_query_field="vector_query_field.predicted_value",
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(query_model_id="sentence-transformers__all-minilm-l6-v2")
    )


if __name__ == '__main__':
    # create index
    # vector_store = init_es_vectorstore_with_hosted_model()
    # create_index_with_vector_field(index_name, index_mapping)
    # load data into es
    try:
        original_dataset = open(DATASET)
        index_passages_into_es(original_dataset)
    except Exception as e:
        print(e)
        exit(1)
