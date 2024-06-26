import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense


# Define the sentences and labels
sentences = ["I am good", "I feel bad"]
labels = [0, 1]

# Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# Pad the sequences to ensure uniform input length
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Convert labels to numpy array
labels = np.array(labels)

# Build the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=max_length))
model.add(LSTM(units=8))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, labels, epochs=10)

# Example prediction
test_sentence = ["I feel good"]
test_sequence = tokenizer.texts_to_sequences(test_sentence)
test_padded_sequence = pad_sequences(test_sequence, maxlen=max_length)
prediction = model.predict(test_padded_sequence)

print(f"Prediction for '{test_sentence[0]}': {prediction[0][0]}")