from flask import Flask, request, jsonify
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and TF-IDF vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    message = preprocess_text(data['message'])
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]
    result = 'spam' if prediction == 1 else 'ham'
    
    return jsonify({'message': data['message'], 'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
