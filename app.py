import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.keras', compile=False)

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    if not text.strip():
        return "Please enter some text."

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return "Input words are not in vocabulary. Try different words."

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]  # Trim to match input length

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]  # [0] to get the scalar value

    # Reverse lookup in tokenizer word_index
    predicted_word = None
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            predicted_word = word
            break

    if predicted_word:
        return predicted_word
    else:
        return "Could not find a matching word. Try again."

# Streamlit UI
st.title("🧠 Next Word Prediction using LSTM")

input_text = st.text_input("Enter a sequence of words:", "To be or not to")

if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # input_length used during training
    result = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f"**Next word:** `{result}`")
