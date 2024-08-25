# LLM Foundations: Vector Databases for Caching and Retrieval Augmented Generation (RAG)

* SOURCE: <https://www.linkedin.com/learning/llm-foundations-vector-databases-for-caching-and-retrieval-augmented-generation-rag/set-up-milvus-and-exercise-files?autoSkip=true&contextUrn=urn%3Ali%3AlearningCollection%3A7225532037028274176&resume=false&u=0>

## Set up Milvus and exercise files

* Ensure Docker is running

* In the terminal:

```sh
conda activate pytorch-stable-pip
cd git/HorusAlkebulan/mle-learning-python/llm_foundations_vector_db_rag
docker-compose -f milvus-standalone-docker-compose.yml up -d
```

* Once complete, ensure all containers are running:

```sh
docker ps
```

* In a browser, open the web interface: <http://localhost:8000/#/connect>.

