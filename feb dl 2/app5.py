import numpy as np
import pickle
import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------
# Load Saved Files
# ------------------------
model = load_model("gru_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

max_length = 50

# ------------------------
# Prediction Function
# ------------------------
def predict_tweet(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post')
    prediction = model.predict(padded)
    class_index = np.argmax(prediction)
    label = label_encoder.inverse_transform([class_index])[0]
    
    if label == 0:
        return "Hate Speech"
    elif label == 1:
        return "Offensive Language"
    else:
        return "Neither"

# ------------------------
# Gradio Interface
# ------------------------
interface = gr.Interface(
    fn=predict_tweet,
    inputs=gr.Textbox(lines=3, placeholder="Enter a tweet here..."),
    outputs="text",
    title="Tweet Hate Speech Detection (GRU Model)",
    description="Enter a sentence to classify it."
)

interface.launch()