import joblib
import re
import numpy as np

print("Loading AITA Ensemble Model...")
try:
    model = joblib.load('aita_model.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: 'aita_model.pkl' not found. Please run train.py first.")
    exit()

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[\[\]\(\)\d]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


print("\n" + "="*40)
print("      AITA JUDGEMENT BOT (ENSEMBLE)      ")
print("="*40)
print("Type your situation below. Type 'quit' to exit.\n")

while True:
    user_input = input("Your Story: ")
    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        break
    if len(user_input.strip()) < 5:
        print(">> Please enter a longer story.")
        continue

    clean_input = clean_text(user_input)
    try:
        probs = model.predict_proba([clean_input])[0]
        prob_nta = probs[0]
        prob_yta = probs[1]
        
        if prob_yta > 0.5:
            verdict = "YTP (You're the Problem)"
            confidence = prob_yta
        else:
            verdict = "NTP (You're Not the Problem)"
            confidence = prob_nta

        print("-" * 30)
        print(f"Verdict:    {verdict}")
        print(f"Confidence: {confidence:.1%} sure")
        print("-" * 30 + "\n")

    except Exception as e:
        print(f"Error during prediction: {e}")

