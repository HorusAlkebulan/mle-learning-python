# ch06_rag_with_milvus.py
import time
from pymilvus import (
    connections,
    db,
    Role,
    utility,
    CollectionSchema,
    FieldSchema,
    DataType,
    Collection,
)
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd

CONNECTION_NAME = "rag_conn"
COLLECTION_NAME = "llm_cache"
DB_NAME = "rag_db"
FIELD_CHUNK_ID = "chunk_id"
FIELD_RAG_TEXT = "rag_text"
FIELD_RAG_EMBEDDING = "rag_embedding"
FIELD_RESPONSE_TEXT = "response_text"
FILENAME_PDF_DOC = "Large Language Models.pdf"
HR = "-------------------------------------------------------------------------"
OPENAI_EMBEDDING_SIZE = 1536
# https://platform.openai.com/docs/deprecations
OPENAI_LLM_MODEL = "gpt-3.5-turbo-instruct" # DEPRECATED: "text-davinci-003"
OPENAI_KEY_ENV_NAME = "OPENAI_API_KEY"
PROJECT_ROOT = os.path.dirname(__file__)


if __name__ == "__main__":

    print(HR)
    print("Setting up the Milvus Cache")
    print(HR)

    connections.add_connection(
        rag_conn={
            "host": "localhost",
            "port": "19530",
            "username": "username",
            "password": "password",
        }
    )

    connections.connect(CONNECTION_NAME)
    current_cxns = connections.list_connections()
    print(f"current connections: {current_cxns}")

    # create db if not already existing
    current_dbs = db.list_database(using=CONNECTION_NAME)
    if DB_NAME not in current_dbs:
        print(f"Creating DB {DB_NAME}")
        db.create_database(
            DB_NAME, using=CONNECTION_NAME
        )  # NOTE: default db is default
    else:
        print(f"DB {DB_NAME} already exists.")

    # switch to this db
    db.using_database(DB_NAME, using=CONNECTION_NAME)

    # set up schema
    chunk_id_field = FieldSchema(name=FIELD_CHUNK_ID, dtype=DataType.INT64, is_primary=True, max_length=32)
    rag_text_field = FieldSchema(name=FIELD_RAG_TEXT, dtype=DataType.VARCHAR, max_length=2048)
    rag_embedding_field = FieldSchema(name=FIELD_RAG_EMBEDDING, dtype=DataType.FLOAT_VECTOR, dim=OPENAI_EMBEDDING_SIZE)

    schema_fields = [chunk_id_field, rag_text_field, rag_embedding_field]
    rag_schema = CollectionSchema(schema_fields, "RAG Schema", enable_dynamic_field=True)
    rag_collection = Collection(COLLECTION_NAME, rag_schema, CONNECTION_NAME, shard_num=2)
    print(f"rag_schema: {rag_collection.schema}")

    print(HR)
    print("### 06.02. Preparing data for Knowledge Base")
    print(HR)

    pdf_path = os.path.join(PROJECT_ROOT, FILENAME_PDF_DOC)
    print(f"Loader: {pdf_path}")
    loader = PDFMinerLoader(pdf_path)
    pdf_docs = loader.load()

    # split doc into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=32,
        length_function=len,
    )
    pdf_docs = text_splitter.split_documents(pdf_docs)

    # turn chunk into list
    rag_text_chunks = []
    for doc in pdf_docs:
        rag_text_chunks.append(doc.page_content)

    print(f"Total chunks: {len(rag_text_chunks)}")
    print(f"Sample chunk[1]: {rag_text_chunks[1]}")

    # initialize openai api key and embeddings model
    openai_api_key = os.environ.get(OPENAI_KEY_ENV_NAME)
    print(f"OPENAI_API_KEY initialized to: {openai_api_key}")

    # create embeddings for all the chunks
    embeddings_model = OpenAIEmbeddings()
    rag_embeddings = []
    for i, chunk in enumerate(rag_text_chunks):
        print(f"- {i}: Computing embedding for {str(chunk)[:100]}...")
        rag_embeddings.append(embeddings_model.embed_query(chunk))
    record_ids = [i for i in range(len(rag_text_chunks))]

    print(HR)
    print("### 06.03. Populating the Milvus database")
    print(HR)

    insert_data = [record_ids, rag_text_chunks, rag_embeddings]
    insert_collection = Collection(COLLECTION_NAME, using=CONNECTION_NAME)
    insert_results = insert_collection.insert(insert_data)
    # flush to commit
    insert_collection.flush()

    # build index
    if insert_collection.has_index():
        print("Index already created for insert_collection")
    else:
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": 1024,
            }
        }    
        insert_collection.create_index(
            field_name=FIELD_RAG_EMBEDDING,
            index_params=index_params,
        )
        index_status = utility.index_building_progress(COLLECTION_NAME, using=CONNECTION_NAME)

    print("Knowledgebase ready.")

    print(HR)
    print("### 06.04 Answering questions with RAG")
    print(HR)

    search_threshold = 0.5
    search_params = {
        "metric_type": "L2",
        "offset": 0,
        "ignore_growing": False,
        "params": {
            "nprobe": 20, 
            "radius": search_threshold,
        }
    }
    query = "What is gender bias?"
    search_embed = embeddings_model.embed_query(query)

    query_collection = Collection(COLLECTION_NAME, using=CONNECTION_NAME)
    query_collection.load()

    query_results = query_collection.search(
        data=[search_embed],
        anns_field=FIELD_RAG_EMBEDDING,
        param=search_params,
        limit=3,
        expr=None,
    )
    top_result = query_results[0][0]
    print(f"top_result: {top_result}")

    # build LLM prompt using results as context
    context = []
    for i in range(len(query_results[0])):
        query_result = query_results[0][i].entity.get(FIELD_RAG_TEXT)
        context.append(query_result)

    augmented_prompt = (f"Based on only the context provided, answer the query below: Context: {str(context)}\n\n Query: {query}")
    print(f"augmented_prompt: {augmented_prompt}")

    print(f"Loading LLM as: {OPENAI_LLM_MODEL}")
    llm_model = OpenAI(temperature=0.0, model=OPENAI_LLM_MODEL)

    llm_result = llm_model(augmented_prompt)
    print(f"llm_result: {llm_result}")