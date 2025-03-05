import json
import torch  
from sentence_transformers import SentenceTransformer, util

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load knowledge base
with open("intents.json", "r") as f:
    intents = json.load(f)

# Create a list of all patterns and corresponding responses
faq_patterns = []
faq_responses = {}

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        faq_patterns.append(pattern)
        faq_responses[pattern] = intent["responses"]

# Compute embeddings for the patterns
faq_embeddings = torch.tensor(embedding_model.encode(faq_patterns))

def get_best_match(user_query):
    """Find the most relevant FAQ response using semantic similarity."""
    query_embedding = torch.tensor(embedding_model.encode([user_query]))
    similarity_scores = util.pytorch_cos_sim(query_embedding, faq_embeddings)
    best_match_idx = similarity_scores.argmax()
    
    if similarity_scores[0][best_match_idx] > 0.7:
        return faq_responses[faq_patterns[best_match_idx]]
    
    return None  # No good match found
