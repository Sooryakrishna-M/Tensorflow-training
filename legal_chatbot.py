import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import json
import sqlite3

# Sample dataset (Can be expanded)
data = {
    "intents": [
        {"tag": "greeting", "patterns": ["Hello", "Hi", "Hey"], "responses": ["Hello! How can I assist you?"]},
        {"tag": "law_query", "patterns": ["What is contract law?", "Explain contract law"], "responses": ["Contract law deals with agreements between parties."]},
        {"tag": "goodbye", "patterns": ["Bye", "Goodbye"], "responses": ["Goodbye! Have a great day."]}
    ]
}

# Extracting data
patterns, tags = [], []
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

tfidf = TfidfVectorizer()
x_train = tfidf.fit_transform(patterns).toarray()

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(tags)

# Building the model
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(x_train.shape[1],), activation='relu'),
    keras.layers.Dense(10, activation='relu'),
    keras.layers.Dense(len(set(tags)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, verbose=0)

# Database setup
conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_input TEXT,
        bot_response TEXT
    )
""")
conn.commit()

# Chat function
def chatbot_response(text):
    x_input = tfidf.transform([text]).toarray()
    prediction = model.predict(x_input)
    tag = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    
    for intent in data['intents']:
        if intent['tag'] == tag:
            response = np.random.choice(intent['responses'])
            
            # Store conversation
            cursor.execute("INSERT INTO history (user_input, bot_response) VALUES (?, ?)", (text, response))
            conn.commit()
            return response
    return "Sorry, I don't understand."

# Test the chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        break
    print("Bot:", chatbot_response(user_input))

conn.close()
