# RAG-on-shared-Google-Drive
This is a dockerized version of Google drive Rag which runs on a shared drive. It uses LLAMA2-7b-hf for summarization and sentence-transformers/LaBSE for generating embeddings. The embeddings are stored in ChromaDB vector database.

## Requirements
- You need a CUDA GPU to run this RAG. CUDA drivers of the GPU must be installed on the system to run.
- In the embedding_service directory, you need to place GCP credentials. The files are as follows:
  - Token.pickle
  - Credentials.json
- These files are necessary for google drive authentication. By using these credentials, our program would be able to access the google drive files.
