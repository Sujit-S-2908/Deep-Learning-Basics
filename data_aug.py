import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('titanic.csv')

# Data preprocessing
def preprocess_data(df):
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True, errors='ignore')
    df.fillna({'Age': df['Age'].median(), 'Embarked': df['Embarked'].mode()[0]}, inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    return df

df = preprocess_data(df)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Normalize numerical features
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation with SMOTE
smote = SMOTE()
X_train, y_train = smote.fit_resample(X_train, y_train)

# Adding Gaussian Noise
def add_noise(X, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

X_train_noisy = add_noise(X_train)

# Define the FCN model
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

# Adaptive Learning Rate: ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# Compile and train model
model = build_model(X_train.shape[1])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train_noisy, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[reduce_lr])

# Evaluate model
eval_results = model.evaluate(X_test, y_test)
print(f'Test Loss: {eval_results[0]}, Test Accuracy: {eval_results[1]}')
