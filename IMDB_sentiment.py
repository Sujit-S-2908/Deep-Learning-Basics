import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load IMDb dataset
df = pd.read_csv("IMDB Dataset.csv")

# Convert sentiment labels to numeric (positive: 1, negative: 0)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Tokenization and padding
max_words = 20000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['review'])
sequences = tokenizer.texts_to_sequences(df['review'])
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['sentiment'].values, test_size=0.2, random_state=42)

# Define sentiment analysis model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test))

# Function to classify user input review
def classify_review():
    while True:
      user_review = input("Enter a movie review (or 'exit' to quit): ")
      if user_review.lower() == 'exit':
        break
      review_sequence = tokenizer.texts_to_sequences([user_review])
      review_padded = pad_sequences(review_sequence, maxlen=max_len)
      prediction = model.predict(review_padded)[0][0]
      sentiment = "Positive" if prediction > 0.5 else "Negative"
      print(f"Predicted Sentiment: {sentiment}")

# Ask user for a review and classify
classify_review()