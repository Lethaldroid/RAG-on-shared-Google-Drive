from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from langchain.embeddings import HuggingFaceEmbeddings
import torch
import chromadb

CHROMA_PATH = "chroma"  # Or your desired path

client = chromadb.HttpClient(host="localhost", port=8000) 

def get_embedding_function():
    # Initialize HuggingFace embeddings with the LaBSE model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    return embeddings

# Set up BitsAndBytesConfig for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

# Load the LLaMA 2 tokenizer and 4-bit quantized model
model_name = 'meta-llama/Llama-2-7b-hf'
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
# Define constants
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def query_rag(query_text: str):
    # Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function, client=client)

    # Search the DB for relevant context
    results = db.similarity_search_with_score(query_text, k=5)

    # Format the context from the retrieved documents
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Tokenize and process the prompt
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(model.device)

    # Generate response using the model with fine-tuned settings
    response_ids = model.generate(
        input_ids,
        max_new_tokens=200,  # Adjust this to control the length of the output
        top_k=50,            # Consider only top 50 tokens during sampling
        top_p=0.9,           # Use nucleus sampling
        do_sample=True       # Enable sampling
    )
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # Gather metadata and return results
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text, {"sources": sources}

input_query = "Data data everywhere"
response, metadata = query_rag(input_query)