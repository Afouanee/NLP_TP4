#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust decoder that:
 - uses model.inputs[0] to get encoder input tensor
 - re-applies encoder Embedding before calling encoder LSTM
 - reuses decoder Embedding + LSTM + Dense for one-step decoding
 - performs greedy decoding
"""

import pickle
import sys
import numpy as np
from tensorflow import keras

units = 512
model_file = f"en2fr{units}.keras"
voc_file   = "voc.pkl"
begin = '\x02'
end   = '\x03'
max_target_len = 200

# -----------------------
# Load model and vocab
# -----------------------
print(f"Loading model: {model_file}")
model = keras.models.load_model(model_file)
model.summary()

with open(voc_file, 'rb') as f:
    voc, char2num, num2char = pickle.load(f)
print("Vocabulary loaded:", [len(v) for v in voc])


# Encoder
enc_emb_layer = model.get_layer('l_enc_embedding')
enc_lstm_layer = model.get_layer('l_enc_lstm')

# Decoder
dec_emb_layer = model.get_layer('l_dec_embedding')
dec_lstm_layer = model.get_layer('l_dec_lstm')
dec_dense_layer = model.get_layer('l_dec_dense')


# -----------------------
# Build encoder model: enc_input -> [h, c]
# -----------------------
# Apply encoder embedding to the encoder input tensor, then call the trained LSTM on it
enc_input_tensor = keras.Input(shape=(None,), dtype='int32')
enc_embedded = enc_emb_layer(enc_input_tensor)   # (batch, timesteps, emb_dim)
enc_out, enc_h, enc_c = enc_lstm_layer(enc_embedded)
encoder_model = keras.Model(inputs=enc_input_tensor, outputs=[enc_h, enc_c])
print("Encoder model built.")

# -----------------------
# Build decoder model for one-step decoding
# -----------------------
state_size = dec_lstm_layer.units

dec_state_input_h = keras.Input(shape=(state_size,), name='dec_state_input_h')
dec_state_input_c = keras.Input(shape=(state_size,), name='dec_state_input_c')
dec_input_single = keras.Input(shape=(1,), dtype='int32', name='dec_input_single')  # a single timestep token

# apply decoder embedding then decoder LSTM (one step)
dec_embedded = dec_emb_layer(dec_input_single)  # (batch, 1, emb_dim)
dec_outputs, dec_h, dec_c = dec_lstm_layer(dec_embedded, initial_state=[dec_state_input_h, dec_state_input_c])
dec_outputs = dec_dense_layer(dec_outputs)  # (batch, 1, vocab_size)

decoder_model = keras.Model(
    [dec_input_single, dec_state_input_h, dec_state_input_c],
    [dec_outputs, dec_h, dec_c]
)
print("Decoder model built.")

# -----------------------
# Utilities: encode input / greedy decode
# -----------------------
def encode_input(text: str):
    """Encode source text into int32 array (1, L). Append end marker if missing."""
    if not text.endswith(end):
        text = text + end
    arr = np.zeros((1, len(text)), dtype='int32')
    for i, ch in enumerate(text):
        arr[0, i] = char2num[0].get(ch, 0)
    return arr

def decode_greedy(src_text: str, maxlen: int = max_target_len):
    enc_seq = encode_input(src_text)
    # get encoder states
    h, c = encoder_model.predict(enc_seq, verbose=0)

    # start token (<begin>)
    cur_token = np.array([[char2num[1][begin]]], dtype='int32')
    decoded_chars = []

    for _ in range(maxlen):
        out_tokens, h, c = decoder_model.predict([cur_token, h, c], verbose=0)
        probs = out_tokens[0, 0, :]
        idx = int(np.argmax(probs))
        ch = num2char[1][idx]
        if ch == end:
            break
        decoded_chars.append(ch)
        cur_token = np.array([[idx]], dtype='int32')

    return "".join(decoded_chars)

# -----------------------
# Interactive loop
# -----------------------
print("Ready. Enter a source sentence (empty to quit).")
try:
    while True:
        s = input("Source > ").strip()
        if s == "":
            break
        translation = decode_greedy(s)
        print("→", translation)
except KeyboardInterrupt:
    print("\nExiting.")
    sys.exit(0)

