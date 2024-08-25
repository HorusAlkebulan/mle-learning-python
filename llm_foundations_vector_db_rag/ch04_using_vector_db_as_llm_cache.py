# ch04_using_vector_db_as_llm_cache.py

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
import pandas as pd

COLLECTION_NAME = "llm_cache"
CONNECTION_NAME = "cache_conn"
DB_NAME = "cache_db"
FIELD_PROMPT_EMBEDDING = "prompt_embedding"
FIELD_PROMPT_TEXT = "prompt_text"
FIELD_RESPONSE_TEXT = "response_text"
HR = "-------------------------------------------------------------------------"
OPENAI_EMBEDDING_SIZE = 1536
OPENAI_LLM_MODEL = "gpt-3.5-turbo-instruct" # DEPRECATED: "text-davinci-003"
OPENAI_KEY_ENV_NAME = "OPENAI_API_KEY"
PROJECT_ROOT = os.path.dirname(__file__)


def get_response(
    prompt_text: str,
    embeddings_model: OpenAIEmbeddings,
    cache_collection: Collection,
    search_params: dict,
    llm_model: OpenAI,
) -> str:

    start_time = time.time()

    prompt_embed = embeddings_model.embed_query(prompt_text)

    # check cache
    cache_results = cache_collection.search(
        data=[prompt_embed],
        anns_field=FIELD_PROMPT_EMBEDDING,
        param=search_params,
        limit=1,
        expr=None,
        output_fields=[FIELD_PROMPT_TEXT, FIELD_RESPONSE_TEXT],
        consistency_level="Strong",
    )
    returned_response = "None"

    if (len(cache_results[0]) > 0):
        cache_response = cache_results[0][0].entity.get(FIELD_RESPONSE_TEXT)
        print(f"Cache Hit --> prompt: {prompt_text}, cache_response: {cache_response}")
        returned_response = cache_response
    else:
        llm_response = llm_model(prompt_text)
        print(f"Cache Miss --> prompt: {prompt_text}, llm_response: {llm_response}")
        returned_response = llm_response
    
        # save to cache
        cache_prompt = [prompt_text]
        cache_prompt_embedding = [prompt_embed]
        response_text = [llm_response]
        insert_data = [cache_prompt, response_text, cache_prompt_embedding]
        mr = cache_collection.insert(insert_data)
        print(f"mr: {mr}")
    
    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time:0.3f}")
    return returned_response


if __name__ == "__main__":

    print(HR)
    print("Setting up the Milvus Cache")
    print(HR)

    # create connection
    connections.add_connection(
        cache_conn={
            "host": "localhost",
            "port": "19530",
            "username": "username",
            "password": "password",
        }
    )

    # connect
    connections.connect(CONNECTION_NAME)

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

    # define the schema
    cache_id = FieldSchema(
        name="cache_id",
        dtype=DataType.INT64,
        auto_id=True,
        is_primary=True,
        max_length=32,
    )

    prompt_text = FieldSchema(
        name="prompt_text",
        dtype=DataType.VARCHAR,
        max_length=2048,
    )

    response_text = FieldSchema(
        name="response_text",
        dtype=DataType.VARCHAR,
        max_length=2048,
    )

    prompt_embedding = FieldSchema(
        name="prompt_embedding", dtype=DataType.FLOAT_VECTOR, dim=OPENAI_EMBEDDING_SIZE
    )

    # create the collection (table?)
    schema_fields = [cache_id, prompt_text, response_text, prompt_embedding]
    schema_description = "Cache for LLM"
    cache_schema = CollectionSchema(
        schema_fields, schema_description, enable_dynamic_field=True
    )
    cache_collection = Collection(
        COLLECTION_NAME, cache_schema, CONNECTION_NAME, shard_num=2
    )
    print(f"Schema created: {cache_collection.schema}")

    if not cache_collection.has_index():
        print(f"Creating index on {FIELD_PROMPT_EMBEDDING}")
        # build collection index on embedding field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        cache_collection.create_index("prompt_embedding", index_params)

        # flush and persist
        cache_collection.flush()
    else:
        print(f"Index already created for: {FIELD_PROMPT_EMBEDDING}")

    # now load
    cache_collection.load()

    print("Cache Collection Ready.")

    print(HR)
    print("## 04.04. Inference Process with caching")
    print(HR)

    # initialize openai api key and embeddings model
    openai_api_key = os.environ.get(OPENAI_KEY_ENV_NAME)
    print(f"OPENAI_API_KEY initialized to: {openai_api_key}")

    print(f"Loading LLM as: {OPENAI_LLM_MODEL}")
    llm_model = OpenAI(temperature=0.0, model=OPENAI_LLM_MODEL)

    embeddings_model = OpenAIEmbeddings()

    similarity_threshold = 0.3
    search_params = {
        "metric_type": "L2",
        "offset": 0,
        "ignore_growing": False,
        "params": {
            "nprobe": 20,
            "radius": similarity_threshold,
        },
    }

    # build up the cache
    print(HR)
    print("Running queries to build the cache")
    prompt_text = "In which year was Abraham Lincoln born?"
    response = get_response(
        prompt_text, embeddings_model, cache_collection, search_params, llm_model
    )
    prompt_text = "What is distance between the sun and the moon?"
    response = get_response(
        prompt_text, embeddings_model, cache_collection, search_params, llm_model
    )
    prompt_text = "How many years have Lebron James played in the NBA?"
    response = get_response(
        prompt_text, embeddings_model, cache_collection, search_params, llm_model
    )
    prompt_text = "What are the advantages of the python language?"
    response = get_response(
        prompt_text, embeddings_model, cache_collection, search_params, llm_model
    )
    prompt_text = "What is the typical height of an elephant"
    response = get_response(
        prompt_text, embeddings_model, cache_collection, search_params, llm_model
    )

    # Now test with cache
    print(HR)
    print("With cache in place, running queries again.")

    prompt_text = "List some advantages of the python language"
    response = get_response(
        prompt_text, embeddings_model, cache_collection, search_params, llm_model
    )
    prompt_text = "How tall is an elephant?"
    response = get_response(
        prompt_text, embeddings_model, cache_collection, search_params, llm_model
    )
