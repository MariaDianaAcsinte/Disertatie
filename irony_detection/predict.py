from keras.models import load_model
from numpy import concatenate

model = load_model('simple_text_model.h5')

# Importing all the necessary libraries
import keras
import numpy as np
from keras import optimizers
from keras.models import load_model
from keras.layers import Bidirectional
from keras.src.layers import GlobalAveragePooling2D
from keras.models import Sequential

from Multimodal_baseline_Functions import *
from keras.layers import Reshape, Dropout
from keras.utils import plot_model
import os
import matplotlib.pyplot as plt
from keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling3D
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras import regularizers
from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array



# Function to plot confusion matrix using Matplotlib
def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm, cmap=plt.cm.Greens)
    fig.colorbar(cax)

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', color='red')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()



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
    for t, m in zip(text, mask):
        if m:
            new_text.append(t)

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

maxlen = max(len(seq) for seq in sequences_train)

x_train = preprocessing.sequence.pad_sequences(sequences_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(sequences_test, maxlen=maxlen)
x_val = preprocessing.sequence.pad_sequences(sequences_val, maxlen=maxlen)

# encoding all the labels
y_test = np.array(testing_DF['label'])[mask_test]
y_train = np.array(training_DF['label'])[mask_train]
y_val = np.array(validation_DF['label'])[mask_val]


pred_val = model.predict(x_val)
pred_val = (pred_val > 0.5).astype(int).flatten()

from sklearn.metrics import accuracy_score, precision_score, \
    recall_score, f1_score, matthews_corrcoef, confusion_matrix

#accuracy = accuracy_score(y_val, pred_val)
#print(accuracy)
precision = precision_score(y_val, pred_val)
print(f"Precision: {precision}")
recall = recall_score(y_val, pred_val)
print(f"Recall: {recall}")
f1 = f1_score(y_val, pred_val)
print(f"F1: {f1}")

cm1 = confusion_matrix(y_val, pred_val)
print(cm1)
tn, fp, fn, tp = cm1.ravel()

# Labels for the confusion matrix
labels = ['Non-Ironic', 'Ironic']

# Plotting the confusion matrix
plot_confusion_matrix(cm1, labels)
