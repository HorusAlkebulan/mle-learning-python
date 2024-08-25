# ch04_using_vector_db_as_llm_cache.py

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
import pandas as pd

COLLECTION_NAME = "llm_cache"
CONNECTION_NAME = "cache_conn"
DB_NAME = "cache_db"
FIELD_PROMPT_EMBEDDING = "prompt_embedding"
HR = "-------------------------------------------------------------------------"
OPENAI_EMBEDDING_SIZE = 1536
OPENAI_KEY_ENV_NAME = "OPENAI_API_KEY"
PROJECT_ROOT = os.path.dirname(__file__)


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
        name="prompt_embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=OPENAI_EMBEDDING_SIZE
    )

    # create the collection (table?)
    schema_fields = [cache_id, prompt_text, response_text, prompt_embedding]
    schema_description = "Cache for LLM"
    cache_schema = CollectionSchema(schema_fields, schema_description, enable_dynamic_field=True)
    cache_collection = Collection(COLLECTION_NAME, cache_schema, CONNECTION_NAME, shard_num=2)
    print(f"Schema created: {cache_collection.schema}")

    if not cache_collection.has_index():
        print(f"Creating index on {FIELD_PROMPT_EMBEDDING}")
        # build collection index on embedding field
        index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}}
        cache_collection.create_index("prompt_embedding", index_params)

        # flush and persist
        cache_collection.flush()
    else:
        print(f"Index already created for: {FIELD_PROMPT_EMBEDDING}")

    # now load
    cache_collection.load()

    print("Cache Collection Ready.")
