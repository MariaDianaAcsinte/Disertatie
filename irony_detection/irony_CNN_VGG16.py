# Importing all the necessary libraries
import keras
import numpy as np
from keras import optimizers
from keras.models import load_model
from keras.layers import Bidirectional
from keras.src.layers import GlobalAveragePooling2D
from keras.models import Sequential
#from keras.src.layers import Dense, concatenate
from keras.models import Model, load_model
#from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.src.metrics import Precision, Recall
from keras_preprocessing.sequence import pad_sequences

from Multimodal_baseline_Functions import *
from keras.layers import Reshape, Dropout
from keras.utils import plot_model
import os
import matplotlib.pyplot as plt
from keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling3D
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from keras import regularizers
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import concatenate
from keras import layers
from keras import regularizers
from keras_preprocessing.image import load_img, img_to_array

layer = layers.Dense(
    units=64,
    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.L2(1e-4),
    activity_regularizer=regularizers.L2(1e-5)
)

# Assigning class weights
class_weight = {1: 1.4,
                0: 1.}

GLOVE_DIR = "E:\\Master\\Disertatie\\DISERATATIE\\glove.6B"
EMBEDDING_DIM = 50
num_epochs = 7
step_epochs = 2
val_steps = 149

# Defining model with Adam optimizer
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = optimizers.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adadelta = optimizers.Adadelta(learning_rate=1.0, rho=0.95, epsilon=None, decay=0.0)


def Image_model(base_model):
    # Freezing all the trainable layers
    for layer in base_model.layers:
        layer.trainable = False

    # Creating output layer
    x = base_model.output
    # Adding pooling layer before the output
    x = GlobalAveragePooling2D()(x)
    # Adding a fully-connected layer
    # x = Dense(1024, activation='relu')(x)
    # and a logistic layer with 2 classes
    return x


def read_data(file_name):
    # Opening file
    with open(file_name, 'r', encoding="utf8") as f:
        # Creating empty set and dictonary for vocab and word respectively
        word_vocab = set()
        word2vector = {}
        # Iterating over each line of file
        for line in f:
            # Spliting lines
            line_ = line.strip()
            # Splitting words
            words_Vec = line_.split()
            word_vocab.add(words_Vec[0])
            word2vector[words_Vec[0]] = np.array(words_Vec[1:], dtype=float)
    print("Total Words in DataSet:", len(word_vocab))
    return word_vocab, word2vector


# Dividing data in test, train, validation
training_DF, testing_DF, validation_DF = preprocess_text(Training_path, Validation_path, Testing_path)

# Processing image and text for each set
# Creating train, test and validation image path
train_img_path = create_img_path(training_DF, 'image_name', img_dir)
test_img_path = create_img_path(testing_DF, 'image_name', img_dir)
val_img_path = create_img_path(validation_DF, 'image_name', img_dir)

# Processing the text
training_DF['sentence'] = training_DF['sentence'].apply(clean_text)
testing_DF['sentence'] = testing_DF['sentence'].apply(clean_text)
validation_DF['sentence'] = validation_DF['sentence'].apply(clean_text)

# Vectorising text
# process the whole observation into single list
train_text_list = list(training_DF['sentence'])
test_text_list = list(testing_DF['sentence'])
val_text_list = list(validation_DF['sentence'])

mask_train = np.array([len(t) < 300 for t in train_text_list])
mask_test = np.array([len(t) < 300 for t in test_text_list])
mask_val = np.array([len(t) < 300 for t in val_text_list])

def apply_mask(text, mask):
    new_text = []
    count = 1
    for t, m in zip(text, mask):
        if m:
            new_text.append(t)
        # else:
            # print(count)
            # print(t)
        count += 1

    return new_text
train_text_list = apply_mask(train_text_list, mask_train)
test_text_list = apply_mask(test_text_list, mask_test)
val_text_list = apply_mask(val_text_list, mask_val)

all_list = train_text_list + test_text_list + val_text_list

# Creating vectors for train, test, validation
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_list)
sequences_train = tokenizer.texts_to_sequences(train_text_list)
sequences_test = tokenizer.texts_to_sequences(test_text_list)
sequences_val = tokenizer.texts_to_sequences(val_text_list)

#vocab_size = len(tokenizer.word_index) + 1

maxlen = max(len(seq) for seq in sequences_train)

x_train = preprocessing.sequence.pad_sequences(sequences_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(sequences_test, maxlen=maxlen)
x_val = preprocessing.sequence.pad_sequences(sequences_val, maxlen=maxlen)

# encoding all the labels
y_test = np.array(testing_DF['label'])[mask_test]
y_train = np.array(training_DF['label'])[mask_train]
y_val = np.array(validation_DF['label'])[mask_val]

vocab, w2v = read_data(os.path.join(GLOVE_DIR, "glove.6B.50d.txt"))

word_index = tokenizer.word_index
num_tokens = len(word_index)
vocab_size = len(tokenizer.word_index) + 1

max_sequence_length = max(len(seq) for seq in sequences_train)

# Creating embeddding weight matrix
embedding_matrix = np.zeros((num_tokens + 1, EMBEDDING_DIM))

for word, i in word_index.items():
    embedding_vector = w2v.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Creating embedded layer using embedded matrix as weight matrix

embedding_layer = Embedding(
    input_dim=num_tokens + 1, output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix],
    trainable=False
)


# Define the text model using functional API
txt_input = Input(shape=(maxlen,))
txt_embedding = embedding_layer(txt_input)
txt_lstm = LSTM(units=8)(txt_embedding)
txt_output = Dense(1, activation='sigmoid', name="dense_txt_out")(txt_lstm)

txt_model = Model(inputs=txt_input, outputs=txt_output)

# Compile the model
txt_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Plot text model
mask = x_train.sum(-1) != 0
x_train = x_train[mask]
y_train = y_train[mask]

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)

# Training text model
txt_model.fit(x_train, y_train, epochs=500)

txt_model.save('simple_text_model.h5')

history = txt_model.fit(x_train, y_train, epochs=500, validation_data=(x_val, y_val))

# Plotting training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


# Load the pre-trained image model
base_model = VGG16(weights='imagenet', include_top=False)

# Create image model
image_input = Input(shape=(224, 224, 3), name='image_input')
x = base_model(image_input)
x = GlobalAveragePooling2D()(x)
image_output = Dense(1024, activation='relu')(x)
image_model = Model(inputs=image_input, outputs=image_output)

# Load the pre-trained text model
text_model = load_model('simple_text_model.h5')

# Combine the models
combined_input = concatenate([image_model.output, text_model.output])
x = Dense(512, activation='relu')(combined_input)
x = Dropout(0.5)(x)
final_output = Dense(1, activation='sigmoid')(x)

# Final model
combined_model = Model(inputs=[image_model.input, text_model.input], outputs=final_output)

# Compile the model
combined_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])

# Example data paths and labels
training_DF, testing_DF, validation_DF = preprocess_text(Training_path, Validation_path, Testing_path)

# Processing image and text for each set
# Creating train, test and validation image path
def create_img_path(df, column_name, img_dir):
    return [os.path.join(img_dir, img_name) for img_name in df[column_name].tolist()]

# Load and preprocess images
def load_images(image_paths, image_size=(224, 224)):
    loaded_images = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=image_size)
            img = img_to_array(img)
            img = img / 255.0  # Normalizing the image
            loaded_images.append(img)
        else:
            print(f"Image not found: {img_path}")
    return np.array(loaded_images)

# Processing image and text for each set
# Creating train, test and validation image path
train_img_path = create_img_path(training_DF, 'image_name', img_dir)
test_img_path = create_img_path(testing_DF, 'image_name', img_dir)
val_img_path = create_img_path(validation_DF, 'image_name', img_dir)

print(f"Train image paths: {train_img_path[:5]}")
print(f"Validation image paths: {val_img_path[:5]}")

# Processing the text
training_DF['sentence'] = training_DF['sentence'].apply(clean_text)
testing_DF['sentence'] = testing_DF['sentence'].apply(clean_text)
validation_DF['sentence'] = validation_DF['sentence'].apply(clean_text)

# Extract labels from dataframes
train_labels = training_DF['label'].tolist()
val_labels = validation_DF['label'].tolist()
test_labels = testing_DF['label'].tolist()

# from predict import load_and_preprocess_image

def load_and_preprocess_image(image_path, i):
    try:
        if not mask_train[i]:
            return
        image_path = str(image_path)
        # Load the image with the target size
        img = load_img(image_path, target_size=(224, 224))

        # Convert the image to a numpy array
        img_array = img_to_array(img)

        # Normalize the image array to the range [0, 1]
        img_array = img_array / 255.0

        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # print(train_img_path)
        # exit()
        return None

# x_train_img_list = [load_and_preprocess_image(p) for p in train_img_path]
x_train_img_list = [load_and_preprocess_image(p, i) for i, p in enumerate(train_img_path)]

x_train_img = np.array([img for img in x_train_img_list if img is not None])

# x_val_img_list = [load_and_preprocess_image(p) for p in val_img_path]
x_val_img_list = [load_and_preprocess_image(p, i) for i, p in enumerate(val_img_path)]
x_val_img = np.array([img for img in x_val_img_list if img is not None])

print(f"Number of training images loaded: {len(x_train_img)}")
print(f"Number of validation images loaded: {len(x_val_img)}")

# Tokenize and preprocess texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_list)
sequences_train = tokenizer.texts_to_sequences(train_text_list)
sequences_test = tokenizer.texts_to_sequences(test_text_list)
sequences_val = tokenizer.texts_to_sequences(val_text_list)

tokenizer = Tokenizer()
all_list = training_DF['sentence'].tolist() + testing_DF['sentence'].tolist() + validation_DF['sentence'].tolist()
tokenizer.fit_on_texts(all_list)

train_text_list = training_DF['sentence'].tolist()
test_text_list = testing_DF['sentence'].tolist()
val_text_list = validation_DF['sentence'].tolist()

sequences_train = tokenizer.texts_to_sequences(train_text_list)
sequences_test = tokenizer.texts_to_sequences(test_text_list)
sequences_val = tokenizer.texts_to_sequences(val_text_list)

maxlen = max(len(seq) for seq in sequences_train)

x_train_text = pad_sequences(sequences_train, maxlen=maxlen)
x_test_text = pad_sequences(sequences_test, maxlen=maxlen)
x_val_text = pad_sequences(sequences_val, maxlen=maxlen)

# Ensure all datasets have the same number of samples
min_len = min(len(x_train_img), len(x_train_text), len(train_labels), len(x_train))
x_train_img = x_train_img[:min_len]
x_train = x_train[:min_len]
x_train_text = x_train_text[:min_len]
y_train = np.array(train_labels[:min_len])

min_len = min(len(x_val_img), len(x_val_text), len(val_labels), len(x_val))
x_val_img = x_val_img[:min_len]
x_val = x_val[:min_len]
x_val_text = x_val_text[:min_len]
y_val = np.array(val_labels[:min_len])

# Debugging statements to print shapes of arrays
print("Shapes of training data:")
print("x_train_img shape:", x_train_img.shape)
print("x_train_text shape:", x_train_text.shape)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

print("Shapes of validation data:")
print("x_val_img shape:", x_val_img.shape)
print("x_val_text shape:", x_val_text.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)

# Define class weights
class_weight = {1: 1.4,
                0: 1.}

# Training the combined model
history = combined_model.fit(
    [x_train_img, x_train], y_train,
    epochs=100,  # Change the number of epochs as needed
    validation_data=([x_val_img, x_val], y_val),
    class_weight=class_weight
)

# Predict on the validation set
y_pred = combined_model.predict([x_val_img, x_val])
y_pred = np.round(y_pred)

# Calculate precision, recall, and f1 score
precision = Precision()
recall = Recall()

precision.update_state(y_val, y_pred)
recall.update_state(y_val, y_pred)

precision_value = precision.result().numpy()
recall_value = recall.result().numpy()
f1 = f1_score(y_val, y_pred)

print(f"Precision: {precision_value}")
print(f"Recall: {recall_value}")
print(f"F1 Score: {f1}")

# Plotting training and validation loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()