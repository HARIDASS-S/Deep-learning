import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("labeled_data.csv")

# Use tweet column as input
X = df["tweet"].astype(str)

# Encode class labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["class"])

# -----------------------------
# 2. Text Tokenization
# -----------------------------
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)

# Padding sequences
max_length = 50
X_pad = pad_sequences(X_seq, maxlen=max_length, padding='post', truncating='post')

# -----------------------------
# 3. Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Build GRU Model
# -----------------------------
model = Sequential([
    Embedding(input_dim=10000, output_dim=64, input_length=max_length),
    GRU(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')   # 3 classes
])

# -----------------------------
# 5. Compile Model
# -----------------------------
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# -----------------------------
# 6. Train Model
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# -----------------------------
# 7. Evaluate Model
# -----------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# -----------------------------
# 8. Predict New Tweet
# -----------------------------
def predict_tweet(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    prediction = model.predict(padded)
    class_index = np.argmax(prediction)
    return label_encoder.inverse_transform([class_index])[0]





model.save("gru_model.h5")

import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)