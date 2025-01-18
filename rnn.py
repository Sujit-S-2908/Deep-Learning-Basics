import numpy as np
import pandas as pd

# Sample data
data = {'text': ['I love this product!', 'This is the worst experience ever.', 'Absolutely fantastic!', 'Not good at all.', 'I am very happy', 'This is terrible'],
        'label': [1, 0, 1, 0, 1, 0]}  # 1: positive, 0: negative

df = pd.DataFrame(data)

# Initialize labels for certain words
word_labels = {'love': 1, 'worst': 0, 'fantastic': 1, 'not': 0, 'happy': 1, 'terrible': 0}

# Function to replace words with labels
def label_text(text):
    tokens = text.lower().split()
    labels = [word_labels.get(token, -1) for token in tokens]
    return labels

# Apply the function to the text data
df['word_labels'] = df['text'].apply(label_text)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])

# Padding sequences
max_sequence_length = 10
X = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Labels
y = np.array(df['label'])

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM, Dropout

# Model parameters
vocab_size = 5000
embedding_dim = 64
rnn_units = 128

# Build the model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(rnn_units, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.summary()

# Training parameters
batch_size = 32
epochs = 20

# Train the model
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Evaluate on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Classification report
from sklearn.metrics import classification_report
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))

# Sample prediction
sample_texts = ["I am very happy with this!", "This is terrible."]
sample_sequences = tokenizer.texts_to_sequences(sample_texts)
sample_padded = pad_sequences(sample_sequences, maxlen=max_sequence_length)

predictions = model.predict(sample_padded)
print(predictions)
predictions = ["Positive" if p > 0.5 else "Negative" for p in predictions]

for text, prediction in zip(sample_texts, predictions):
    print(f'Text: {text} \nPrediction: {prediction}\n')
