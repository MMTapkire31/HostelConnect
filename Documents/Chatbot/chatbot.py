import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample training data (Questions & Responses)
questions = [
    "hello", "hi", "hey", 
    "how are you", "what's up", "how's it going",
    "your name", "who are you", 
    "what can you do", "tell me something",
    "bye", "goodbye", "see you"
]

responses = [
    "Hello!", "Hi there!", "Hey!",
    "I'm good, thanks for asking!", "Not much, just chatting!", "I'm doing great!",
    "I'm a simple chatbot!", "I'm an AI chatbot created in Python.",
    "I can chat with you and answer simple questions!", "Sure! Did you know Python is named after Monty Python?",
    "Goodbye! Have a great day!", "See you later!", "Take care!"
]

# Create a text classifier pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(questions, responses)

# Chatbot function
def chatbot():
    print("Chatbot: Hello! Type 'exit' to stop the chat.")
    
    while True:
        user_input = input("You: ").lower()
        if user_input == "exit":
            print("Chatbot: Goodbye! Have a great day.")
            break
        
        response = model.predict([user_input])[0]  # Predict response
        print(f"Chatbot: {response}")

# Start chatbot
chatbot()
