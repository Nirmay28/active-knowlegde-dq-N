import requests
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# --- Configuration ---
OLLAMA_API = "http://localhost:11434/api/generate"  # Ollama endpoint
OLLAMA_MODEL = "phi3"  # Model name

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

# --- Prompt Helper ---
def create_debate_prompt(article_text, side="For"):
    """
    Creates a prompt for AI, optionally biasing the first persona based on side.
    """
    return f"""
You are a debate generator. Your task:

1. Extract exactly 3-4 debatable key claims from the text.
2. Generate a short debate (2 exchanges, max 3 sentences each) between:
   - Persona A (supports claims)
   - Persona B (challenges claims)
   
If side is "{side}", make Persona A argue for {side.lower()} in the first turn.

Output ONLY in this format:

Key Claims:
1. [Claim 1]
2. [Claim 2]
3. [Claim 3]

Debate:
Persona A: [Opening statement]
Persona B: [Counter-argument]
Persona A: [Rebuttal]
Persona B: [Final counter-rebuttal]

Article:
{article_text}
"""

# --- Routes ---
@app.route("/generate_debate", methods=["POST"])
def generate_debate():
    data = request.json
    article = data.get("article", "").strip()
    side = data.get("side", "For")

    if not article:
        return jsonify({"error": "No article text provided"}), 400

    prompt = create_debate_prompt(article, side)

    def stream_response():
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True
            }
            response = requests.post(OLLAMA_API, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        if "response" in chunk:
                            yield chunk["response"]
                    except json.JSONDecodeError:
                        continue
        except requests.RequestException as e:
            yield f"Error connecting to Ollama: {e}"

    return app.response_class(stream_response(), mimetype="text/plain")

# Serve frontend
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


if __name__ == "__main__":
    app.run(port=5000, debug=True)
