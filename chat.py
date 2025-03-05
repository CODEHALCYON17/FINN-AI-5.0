import random
import json
import torch
import google.generativeai as genai
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load intents and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents.json", "r") as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Set up Gemini API
genai.configure(api_key="AIzaSyCAGYFXgAB9dO5OD9RYJfRjr3Nke_kFdvg")
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

bot_name = "Babu"

def get_response(user_input):
    """Determine response using NeuralNet or Google Gemini."""
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()].item()

    # If confidence is high, return predefined response
    if prob > 0.75:
        for intent in intents["intents"]:
            if intent["tag"] == tag:
                return random.choice(intent["responses"])

    # If confidence is low, ask Gemini
    else:
        gemini_response = gemini_model.generate_content(user_input)
        return gemini_response.text

# Example testing
if __name__ == "__main__":
    print("Chatbot is running! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = get_response(user_input)
        print(f"{bot_name}: {response}")
