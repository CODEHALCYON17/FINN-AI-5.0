from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import json
import google.generativeai as genai
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import numpy as np
import random
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS to connect frontend and backend

# Load intents and model
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

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Configure Google Gemini API
genai.configure(api_key="AIzaSyCAGYFXgAB9dO5OD9RYJfRjr3Nke_kFdvg")
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

def get_current_date():
    return datetime.today().strftime('%B %d, %Y')  # Format: 'February 18, 2025'

def get_response(user_input):
    """Determine response using NeuralNet or Google Gemini."""
    
    # Check if the user asks for today's date
    if "today's date" in user_input.lower() or "current date" in user_input.lower():
        return f"Today's date is {get_current_date()}"

    # Proceed to check with the neural network and Gemini if date isn't mentioned
    sentence = tokenize(user_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

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
        
        # Check if Gemini's response is outdated (e.g., today's date is Feb 28, 2023)
        if "february 28, 2023" in gemini_response.text.lower():
            return f"Sorry, Gemini gave outdated data. Here's the current date: {get_current_date()}"
        
        return gemini_response.text

@app.route("/chat", methods=["POST"])
def chat():
    """API Endpoint for chatbot interaction"""
    user_input = request.json.get("message", "")
    response = get_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
