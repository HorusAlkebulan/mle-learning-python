# milvus_db_operations.py

from pymilvus import connections, db, Role, utility, CollectionSchema, FieldSchema, DataType, Collection
import os
from langchain.llms import OpenAI
#NOTE: Deprecated --> from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import pandas as pd

OPENAI_EMBEDDING_SIZE = 1536
OPENAI_KEY_ENV_NAME = "OPENAI_API_KEY"
PROJECT_ROOT = os.path.dirname(__file__)

if __name__ == "__main__":

    # create list of connections
    connections.add_connection(
        learn={
            "host": "localhost",
            "port": "19530",
            "username": "",
            "password": "",
        }
    )

    # connect
    connection_id = "learn"
    connections.connect(connection_id)

    # list all connections
    print("Current connections:")
    print(connections.list_connections())

    # list current dbs
    current_dbs = db.list_database(using=connection_id)
    print(f"Current databases:\n{current_dbs}")

    # create db we need, if necessary
    db_name = "course_db"
    if db_name not in current_dbs:
        print(f"Creating database: {db_name}")
        wiki_db = db.create_database(db_name, using=connection_id)
    else:
        print(f"Required database {db_name} already exists.")

    # switch context to db
    db.using_database(db_name, using=connection_id)
    print(f"Using database: {db_name}")

    # create user, if necessary
    new_user = "course_public"
    new_passwd = "password"
    current_users = utility.list_usernames(using=connection_id)
    print(f"current_users: {current_users}")

    if new_user not in current_users:
        print(f"User '{new_user}' not yet created, creating.")
        utility.create_user(new_user, new_passwd, using=connection_id)
    else:
        print(f"User '{new_user}' already created.")

    new_role = "public"
    print(f"Adding user '{new_user}' to role '{new_role}'.")
    public_role = Role(new_role, using=connection_id)
    role_exists = public_role.is_exist()
    print(f"Role exists for {new_role}: {role_exists}")
    public_role.add_user(new_user)
    print("User now in role.")

    # define fields
    course_id = FieldSchema("course_id", DataType.INT64, is_primary=True, max_length=32)
    title = FieldSchema("title", DataType.VARCHAR, max_length=256)
    description = FieldSchema("description", DataType.VARCHAR, max_length=2048)
    desc_embedding = FieldSchema("desc_embedding", DataType.FLOAT_VECTOR, dim=OPENAI_EMBEDDING_SIZE)

    # define schema
    wiki_schema = CollectionSchema(
        fields=[course_id, title, description, desc_embedding],
        description="Courses List",
        enable_dynamic_field=True,
    )
    collection_name = "courses_list"
    print(f"Defining collection: {collection_name}")

    wiki_collection = Collection(name=collection_name, schema=wiki_schema, using=connection_id, shard_num=2)

    collections_list = utility.list_collections(using=connection_id)
    print(f"Current collections: {collections_list}")

    #  set up existing collection into another object
    r_collection = Collection(collection_name, using=connection_id)
    print(f"Another object created using existing collection:\n{r_collection.schema}")

    # load the course data from csv
    course_descriptions_path = os.path.join(PROJECT_ROOT, "course-descriptions.csv")
    print(f"Loading CSV data from: {course_descriptions_path}")
    course_descriptions_df = pd.read_csv(course_descriptions_path)
    print(f"Data loaded:\n{course_descriptions_df.head()}")

    # initialize openai api key and embeddings model
    openai_api_key = os.environ.get(OPENAI_KEY_ENV_NAME)
    print(f"OPENAI_API_KEY initialized to: {openai_api_key}")
    embeddings_model = OpenAIEmbeddings()

    # prep the data for generating embedding vectors
    