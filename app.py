from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re

app = Flask(__name__)
CORS(app) 

model = joblib.load('aita_model.pkl')


def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[\[\]\(\)\d]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    story = data.get('text', '')
    clean_story = clean_text(story)
    probs = model.predict_proba([clean_story])[0]
    prob_yta = float(probs[1]) # Convert numpy float to standard float for JSON

    return jsonify({
            'yta_score': prob_yta,
            'verdict': 'YTA' if prob_yta > 0.4 else 'NTA' # Simple default
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
