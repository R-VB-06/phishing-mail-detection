
Phising Mail Detection: It uses natural language processing (NLP) techniques to identify suspicious content and translate non-English messages into English for consistent analysis.

Technologies Used

- Python
- scikit-learn
- pandas
- langdetect
- deep-translator
- sentence-transformers
- joblib

Installation

git clone https://github.com/your-username/phishing-mail-detection.git
cd phishing-mail-detection
pip install -r requirements.txt


Steps Execution Flow

1. Load Dataset (with `text_combined` and `label` columns) in the project folder   
2. Run `train.py` to load, clean, vectorize, train, and save model as `.pkl` files  
3. If `rf_phishing_model.pkl` and `rf_vectorizer.pkl` already exist, training is skipped automatically  
4. Run `predict.py` and paste email content when prompted  
5. Script detects language using `langdetect`  
6. If input is in English, translation is skipped; otherwise, it's translated to English using `deep-translator`  
7. Loads saved model (`rf_phishing_model.pkl`) and vectorizer (`rf_vectorizer.pkl`)  
8. Vectorizes input and classifies as phishing or safe  
9. Displays result: `-----Phishing-----` or `----Safe----`



