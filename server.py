from flask import Flask, request, jsonify, send_from_directory
import spacy

app = Flask(__name__)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Serve HTML file at "/"
@app.route("/")
def index():
    return send_from_directory('.', 'NLP_project.html')

# NER endpoint
@app.route("/extract", methods=["POST"])
def extract():
    data = request.get_json()
    text = data.get("text", "")

    doc = nlp(text)
    entities = []

    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            entities.append({
                "name": ent.text,
                "type": map_label(ent.label_)
            })

    return jsonify({"entities": entities})

# Map spaCy labels → your KG types
def map_label(label):
    if label == "PERSON":
        return "Person"
    elif label == "ORG":
        return "Organization"
    elif label == "GPE":
        return "Location"
    return "Unknown"

if __name__ == "__main__":
    app.run(debug=True)