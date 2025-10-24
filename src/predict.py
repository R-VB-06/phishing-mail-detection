import joblib
import os
import pandas as pd
import re
from langdetect import detect
from deep_translator import GoogleTranslator

MODEL_PATH = 'rf_phishing_model.pkl'
VEC_PATH = 'rf_vectorizer.pkl'

def is_valid_text(text):
    if len(text.strip()) < 10:
        return False
    word_count = len(text.split())
    if word_count < 2:
        return False
    if not re.search(r'[a-zA-Z\u0900-\u097F\u0B80-\u0BFF]', text):
        return False
    return True

def translate_if_needed(text):
    try:
        lang = detect(text)
        if lang != 'en':
            print(f"Detected language: {lang}. Translating to English...")
            text = GoogleTranslator(source='auto', target='en').translate(text)
            print("****Translation complete****")
        else:
            print("-----Input is already in English. No translation needed-----")
        return text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VEC_PATH):
        raise FileNotFoundError("====Model or vectorizer file not found. Run train_model.py first====")
    print(">>>>Loading model and vectorizer.....")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    print("Artifacts loaded.")
    return model, vectorizer

def classify_email(text, model, vectorizer):
    text = translate_if_needed(text)
    if not text.strip() or not is_valid_text(text):
        return "!!!!Invalid input. Please enter meaningful email content!!!!"
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return "-----Phishing-----" if prediction == 1 else "----Safe----"


def main():
    print("----Paste your email content----")
    sample_email = input("Your input: ").strip()
    try:
        model, vectorizer = load_artifacts()
        result = classify_email(sample_email, model, vectorizer)
        print(f"\nPrediction: {result}")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
