import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_model = genai.GenerativeModel("gemini-pro")

def get_gemini_response(user_query):
    """Use Google Gemini for open-ended questions."""
    prompt = f"""
    You are an AI assistant named FINN AI, designed for IEEE FISAT.
    Use the knowledge base when possible. If no exact match exists, generate a professional response.

    User: {user_query}
    """
    response = gemini_model.generate_content(prompt)
    return response.text
