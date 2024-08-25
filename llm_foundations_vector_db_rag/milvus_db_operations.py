# milvus_db_operations.py

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
# NOTE: Deprecated --> from langchain.llms import OpenAI
from langchain_community.llms import OpenAI

# NOTE: Deprecated --> from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
import pandas as pd

OPENAI_EMBEDDING_SIZE = 1536
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
OPENAI_EMBEDDING_MODEL_DIM = 1024

OPENAI_KEY_ENV_NAME = "OPENAI_API_KEY"
PROJECT_ROOT = os.path.dirname(__file__)
HR = "-------------------------------------------------------------------------"

def delete_objects(collection, collection_name, connection_id, db_name):

    print(HR)
    print("### 03.08. Deleting objects and entities")
    print(HR)

    q_delete = "course_id in [1002]"
    print(f"Running delete using query: {q_delete}")
    delete_result = r_collection.delete(q_delete)
    print(f"delete_result: {delete_result}")

    print(f"Dropping collection: {collection_name}")
    utility.drop_collection(collection_name, using=connection_id)

    print(F"Dropping database: {db_name}")
    db.drop_database(db_name=db_name, using=connection_id)

if __name__ == "__main__":

    print(HR)
    print("### 03.01. Connecting to Milvus")
    print(HR)

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

    print(HR)
    print("### 03.02. Creating databases and users")
    print(HR)

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

    print(HR)
    print("### 03.03. Creating collections")
    print(HR)

    # define fields
    course_id = FieldSchema("course_id", DataType.INT64, is_primary=True, max_length=32)
    title = FieldSchema("title", DataType.VARCHAR, max_length=256)
    description = FieldSchema("description", DataType.VARCHAR, max_length=2048)
    desc_embedding = FieldSchema(
        "desc_embedding", DataType.FLOAT_VECTOR, dim=OPENAI_EMBEDDING_SIZE
    )

    # define schema
    wiki_schema = CollectionSchema(
        fields=[course_id, title, description, desc_embedding],
        description="Courses List",
        enable_dynamic_field=True,
    )
    collection_name = "courses_list"
    print(f"Defining collection: {collection_name}")

    wiki_collection = Collection(
        name=collection_name, schema=wiki_schema, using=connection_id, shard_num=2
    )

    collections_list = utility.list_collections(using=connection_id)
    print(f"Current collections: {collections_list}")

    #  set up existing collection into another object
    r_collection = Collection(collection_name, using=connection_id)
    print(f"Another object created using existing collection:\n{r_collection.schema}")

    print(f"Attempting to load collection: {collection_name}")
    r_collection.load()
    print("r_collection loaded.")

    embeddings_model: OpenAIEmbeddings = None

    if r_collection.has_index():
        print(f"{collection_name} collection already has index, skipping creation.")
    else:

        print(HR)
        print("### 03.04. Inserting data into Milvus")
        print(HR)

        # load the course data from csv
        course_descriptions_path = os.path.join(PROJECT_ROOT, "course-descriptions.csv")
        print(f"Loading CSV data from: {course_descriptions_path}")
        course_descriptions_df = pd.read_csv(course_descriptions_path)
        print(f"Data loaded:\n{course_descriptions_df.head()}")

        # prep the data for generating embedding vectors
        i_course_id = course_descriptions_df["Course ID"].tolist()
        i_title = course_descriptions_df["Title"].tolist()
        i_description = course_descriptions_df["Description"].tolist()

        # initialize openai api key and embeddings model
        openai_api_key = os.environ.get(OPENAI_KEY_ENV_NAME)
        print(f"OPENAI_API_KEY initialized to: {openai_api_key}")
        embeddings_model = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL, dimensions=OPENAI_EMBEDDING_SIZE
        )

        # generate embedding of description
        i_desc_embedding = [embeddings_model.embed_query(i) for i in i_description]

        # format for data input
        insert_data = [i_course_id, i_title, i_description, i_desc_embedding]

        print(f"Inserting data:\n{insert_data}")
        mr = r_collection.insert(insert_data)
        print("Inserted data. Now flushing.")
        r_collection.flush(timeout=180)

        print(HR)
        print("### 03.05. Build an index")
        print(HR)

        # build an index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": 1024
            }
        }
        r_collection.create_index(
            field_name="desc_embedding",
            index_params=index_params,
        )
        status = utility.index_building_progress(collection_name, using=connection_id)
        print(f"index build status:\n{status}")

        r_collection.load()
        print("r_collection loaded.")

    print(HR)
    print("### 03.06. Querying scalar data")
    print(HR)

    query_expr = "course_id == 1001"
    print(f"Running query: {query_expr}")
    q_result = r_collection.query(query_expr, output_fields=["title", "description"])
    print(f"q_result: {q_result}")
    print(f"Result object type: {type(q_result[0])}")
    print(f"Result object: {q_result[0]}")

    query_expr = "(title like 'MLOps%') && (course_id > 1001) "
    print(f"Running query: {query_expr}")
    q_result = r_collection.query(query_expr, output_fields=["title", "description"])
    print(f"q_result: {q_result}")
    print(f"Result object type: {type(q_result[0])}")
    print(f"Result object: {q_result[0]}")

    print(HR)
    print("### 03.07. Searching Vector fields")
    print(HR)

    # NOTE: L1 = Mahattan Distance, L2 = Euclidean Distance
    search_params = {
        "metric_type": "L2",
        "offset": 0,
        "ignore_growing": False,
        "params": {
            "nprobe": 10,
        },
    }

    if embeddings_model is None:
        embeddings_model = OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL, dimensions=OPENAI_EMBEDDING_SIZE
        )

    # create search string and convert to an embedding
    search_string = "machine learning"
    search_embedding = embeddings_model.embed_query(search_string)

    print(f"Running search using string '{search_string}' as embedding:\n{str(search_embedding)[:100]}...")
    s_results = r_collection.search(
        data=[search_embedding],
        anns_field="desc_embedding",
        param=search_params,
        limit=10,
        expr=None,
        output_fields=["title"],
        consistency_level="Strong",
    )
    print(f"Search results (type: {type(s_results[0])}):")
    for i in s_results[0]:
        print(f"- {i.id}\t{str(round(i.distance, 2))}\t{i.entity.get('title')}")

    # create search string (non related) and convert to an embedding
    search_string = "best movies of the year"
    search_embedding = embeddings_model.embed_query(search_string)

    print(f"Running search using string '{search_string}' as embedding:\n{str(search_embedding)[:100]}...")
    s_results = r_collection.search(
        data=[search_embedding],
        anns_field="desc_embedding",
        param=search_params,
        limit=10,
        expr=None,
        output_fields=["title"],
        consistency_level="Strong",
    )
    print(f"Search results (type: {type(s_results[0])}):")
    for i in s_results[0]:
        print(f"- {i.id}\t{str(round(i.distance, 2))}\t{i.entity.get('title')}")

