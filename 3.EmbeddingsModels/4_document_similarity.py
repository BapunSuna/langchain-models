from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about Rohit Sharma"

# Create embeddings for documents
doc_embeddings = embedding.embed_documents(documents)

# Create embedding for query
query_embedding = embedding.embed_query(query)

# Calculate cosine similarity
scores = cosine_similarity(
    [query_embedding],
    doc_embeddings
)[0]

# Find document with highest similarity
index, score = max(
    enumerate(scores),
    key=lambda x: x[1]
)

print("Query:", query)
print("Most relevant document:", documents[index])
print("Similarity score:", score)