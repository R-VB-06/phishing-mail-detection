import pandas as pd
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

MODEL_PATH = 'rf_phishing_model.pkl'
VEC_PATH = 'rf_vectorizer.pkl'

def load_data(filepath, sample_frac=0.3):
    print("== Loading dataset ==")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"!! File not found: {filepath}")
    df = pd.read_csv(filepath)
    print(f"== Dataset loaded successfully ==")
    print("-- Columns in dataset --", df.columns.tolist())
    df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    print(f">> Sampled {len(df)} rows for training")
    return df

def clean_data(df):
    print("-- Cleaning data --")
    if not {'text_combined', 'label'}.issubset(df.columns):
        raise KeyError("!! Required columns 'text_combined' and 'label' not found.")
    df = df.dropna(subset=['label', 'text_combined']).copy()
    df['label'] = df['label'].astype(int)
    print(f"== Data cleaned. Remaining samples: {len(df)} ==")
    return df

def preprocess_data(df):
    print("-- Preprocessing text --")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text_combined'], df['label'], test_size=0.2, random_state=42
    )
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("== Vectorization complete ==")
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer

def train_model(X_train_vec, y_train):
    print("-- Training model --")
    start = time.time()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_vec, y_train)
    print(f"== Model trained in {time.time() - start:.2f} seconds ==")
    return model

def evaluate_model(model, X_test_vec, y_test):
    print("-- Evaluating model --")
    y_pred = model.predict(X_test_vec)
    print(f"== Accuracy: {accuracy_score(y_test, y_pred):.2f} ==")
    print("-- Classification Report --")
    print(classification_report(y_test, y_pred))

def save_artifacts(model, vectorizer):
    print("-- Saving model artifacts --")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    print(f"== Artifacts saved: {MODEL_PATH}, {VEC_PATH} ==")

def model_exists():
    return os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)

if __name__ == "__main__":
    try:
        if model_exists():
            print("!! Model already exists. Skipping training !!")
        else:
            df = load_data('email1.csv', sample_frac=0.3)
            df = clean_data(df)
            X_train_vec, X_test_vec, y_train, y_test, vectorizer = preprocess_data(df)
            model = train_model(X_train_vec, y_train)
            evaluate_model(model, X_test_vec, y_test)
            save_artifacts(model, vectorizer)
            print("== Training complete. Ready for prediction ==")
    except Exception as e:
        print(f"!! Error: {e}")
