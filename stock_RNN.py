import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
file_path = 'TSLA.csv'
data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Select relevant feature (closing price for forecasting)
prices = data[['Close']].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
prices_scaled = scaler.fit_transform(prices)

# Prepare sequences for time series forecasting
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 60  # 60 days of past data to predict next day
X, y = create_sequences(prices_scaled, SEQ_LENGTH)

# Split data into train, validation, test
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)
test_size = len(X) - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# Function to build and train models
def build_and_train_model(model_type):
    model = Sequential()
    
    if model_type == "LSTM":
        model.add(LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
    elif model_type == "GRU":
        model.add(GRU(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)))
        model.add(Dropout(0.2))
        model.add(GRU(50, return_sequences=False))
    elif model_type == "SimpleRNN":
        model.add(SimpleRNN(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(50, return_sequences=False))
    
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stopping], verbose=0)
    
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_actual = scaler.inverse_transform(y_pred)
    y_test_actual = scaler.inverse_transform(y_test)
    
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    accuracy = 100 - (mae / np.mean(y_test_actual)) * 100  # Approximate accuracy
    
    return model_type, test_loss, history, y_test_actual, y_pred_actual, mse, mae, accuracy

# Train and evaluate each model
models = ["LSTM", "GRU", "SimpleRNN"]
results = {}
histories = {}
metrics = {}

plt.figure(figsize=(12,6))
for model_type in models:
    model_name, test_loss, history, y_test_actual, y_pred_actual, mse, mae, accuracy = build_and_train_model(model_type)
    results[model_name] = test_loss
    histories[model_name] = history
    metrics[model_name] = {'MSE': mse, 'MAE': mae, 'Accuracy': accuracy}
    plt.plot(y_pred_actual, label=f'{model_name} Prediction')

plt.plot(y_test_actual, label='Actual Prices', color='black', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.title('Comparison of LSTM, GRU, and SimpleRNN Predictions')
plt.show()

# Plot training loss and accuracy comparison
fig, axs = plt.subplots(2, 1, figsize=(12,12))

for model_name, history in histories.items():
    axs[0].plot(history.history['loss'], label=f'{model_name} Training Loss')
    axs[0].plot(history.history['val_loss'], linestyle='dashed', label=f'{model_name} Validation Loss')
    axs[1].bar(model_name, metrics[model_name]['Accuracy'], label=f'{model_name} Accuracy')

axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()
axs[0].set_title('Training and Validation Loss Comparison')

axs[1].set_ylabel('Accuracy (%)')
axs[1].set_title('Model Accuracy Comparison')
axs[1].legend()

plt.show()

# Print evaluation results
for model_name, metric in metrics.items():
    print(f"{model_name} Test Loss: {results[model_name]}")
    print(f"{model_name} Mean Squared Error (MSE): {metric['MSE']}")
    print(f"{model_name} Mean Absolute Error (MAE): {metric['MAE']}")
    print(f"{model_name} Accuracy: {metric['Accuracy']:.2f}%\n")
