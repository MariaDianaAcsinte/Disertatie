# Importing all the necessary libraries
import keras
import h5py
import learn as learn
from keras import optimizers
from keras.models import load_model
from keras.layers import Bidirectional
from Multimodal_baseline_Functions import *
from keras.layers.core import Reshape, Dropout
from keras.utils.vis_utils import plot_model
import os
import matplotlib.pyplot as plt
from keras.layers import Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling3D
from keras import regularizers
import seaborn as sns
import matplotlib.pyplot as plt
# from scikit.learn import confusion_matrix
from sklearn.metrics import confusion_matrix
from keras.utils import pad_sequences
from keras import regularizers
from keras.applications.inception_v3 import InceptionV3

# Assigning class weights
class_weight = {1: 1.4,
                0: 1.}
GLOVE_DIR = "E:\\Master\\Disertatie\\DISERATATIE\\glove.6B"
EMBEDDING_DIM = 500
num_epochs = 7
step_epochs = 2
val_steps = 149

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

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
  #Opening file
    with open(file_name,'r', encoding="utf8") as f:
      #Creating empty set and dictonary for vocab and word respectively
        word_vocab = set()
        word2vector = {}
        #Iterating over each line of file
        for line in f:
            #Spliting lines
            line_ = line.strip()
            #Splitting words
            words_Vec = line_.split()
            word_vocab.add(words_Vec[0])
            word2vector[words_Vec[0]] = np.array(words_Vec[1:],dtype=float)
    print("Total Words in DataSet:",len(word_vocab))
    return word_vocab,word2vector

# Dividing data in test, train, validation
training_DF, testing_DF, validation_DF = preprocess_text(Training_path,Validation_path, Testing_path)

training_DF.head()

# Processing image and text for each set
# Creating train, test and validation image path
train_img_path = create_img_path(training_DF,'image_name', img_dir)
test_img_path = create_img_path(testing_DF,'image_name', img_dir)
val_img_path = create_img_path(validation_DF,'image_name', img_dir)

# Processing the text
training_DF['sentence'] = training_DF['sentence'].apply(clean_text)
testing_DF['sentence'] = testing_DF['sentence'].apply(clean_text)
validation_DF['sentence'] = validation_DF['sentence'].apply(clean_text)

# Vectorising text
# process the whole observation into single list
train_text_list=list(training_DF['sentence'])
test_text_list = list(testing_DF['sentence'])
val_text_list = list(validation_DF['sentence'])

# Creating vectors for train, test, validation
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(train_text_list)
sequences_train = tokenizer.texts_to_sequences(train_text_list)
sequences_test = tokenizer.texts_to_sequences(test_text_list)
sequences_val = tokenizer.texts_to_sequences(val_text_list)

# preprocessing.sequence.
x_train = pad_sequences(sequences_train, maxlen=maxlen)
x_test = pad_sequences(sequences_test, maxlen=maxlen)
x_val = pad_sequences(sequences_val , maxlen=maxlen)

# encoding all the labels
y_test = testing_DF['label']
y_train = training_DF['label']
y_val = validation_DF['label']

# Creating train, test, val, generator for meme
img_txt_gen_train = img_text_generator(train_img_path, x_train, y_train, batch_size=32)
img_txt_gen_test = img_text_generator(test_img_path, x_test, y_test, batch_size=1)
img_txt_gen_val = img_text_generator(val_img_path, x_val, y_val, batch_size=1)

# Creating train, test, val, generator for text
txt_gen_train = text_generator(x_train, y_train, batch_size=32)
txt_gen_test = text_generator(x_test, y_test, batch_size=1)
txt_gen_val = text_generator(x_val, y_val, batch_size=1)

# Creating train, test, val, generator for image
img_gen_train = image_generator(train_img_path, training_DF, batch_size=32)
img_gen_test = image_generator(test_img_path, testing_DF, batch_size=1)
img_gen_val = image_generator(val_img_path, validation_DF, batch_size=1)

vocab, w2v = read_data(os.path.join(GLOVE_DIR, "glove.6B.50d.txt"))

word_index = tokenizer.word_index
num_tokens = len(word_index)

#Creating embeddding weight matrix
embedding_matrix = np.zeros((num_tokens + 1, EMBEDDING_DIM))

# Ajustați dimensiunea matricei de încorporare la dimensiunea vectorului de încorporare
# EMBEDDING_DIM = embedding_vector.shape[0]  # Actualizați dimensiunea la cea a vectorului de încorporare returnat
# embedding_matrix = np.zeros((num_tokens + 1, EMBEDDING_DIM))


for word, i in word_index.items():
    embedding_vector = w2v.get(word)
    if embedding_vector is not None and embedding_vector.shape[0] == EMBEDDING_DIM:
        embedding_matrix[i] = embedding_vector
    else:
        print(f"Skipping word '{word}' with embedding vector shape {embedding_vector}")


#Creating embedded layer using embedded matrix as weight matrix
embedding_layer = Embedding(num_tokens + 1, EMBEDDING_DIM, weights=[embedding_matrix], trainable = False)

from keras import regularizers

# Defining second LSTM
main_input = Input(shape=(maxlen,), dtype='int32', name='main_input')
# Adding embedding layer
embedded_sequences = embedding_layer(main_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
lstm1 = LSTM(32, return_state=True)
encoder_outputs,state_h,state_c = (lstm1)(x)
states= [state_h,state_c]

# Defining second LSTM
lstm2=LSTM(32, return_sequences=True, return_state=True)
decoder_out,_,_=lstm2(embedded_sequences,initial_state=states)
lstm_out = Flatten()(decoder_out)
txt_out = Dense(1, activation='sigmoid')(lstm_out)

# Defining text model
txt_model = Model(inputs = [main_input], outputs=txt_out)

# Compile text model
txt_model.compile(loss='binary_crossentropy', optimizer=adam, metrics = ["accuracy"])

# Plot text model
plot_model(txt_model, to_file='CNN_txt_model.png', show_shapes=True, show_layer_names=True)

# Training text model
txt_model.fit_generator(txt_gen_train, epochs = num_epochs, validation_steps = val_steps, steps_per_epoch=step_epochs, validation_data=txt_gen_val, shuffle = False, class_weight=class_weight)

# Saving text model
txt_model.save('CNN_txt_model.h5')

# Plotting training and validation loss
loss_values = txt_model.history.history['loss']
val_loss_values = txt_model.history.history['val_loss']
epochs = range(1, 7 + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predicting labels using text model
y_pred_txt = (txt_model.predict_generator(txt_gen_test,steps = 149))
y_pred_txt = np.round(list(itertools.chain(*y_pred_txt)))
y_true = y_test.values

# Confusion matrix
labels = [1,0]
cm = confusion_matrix(y_true, y_pred_txt, labels=labels)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, cmap='Greens'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['ironic', 'non-ironic']); ax.yaxis.set_ticklabels(['ironic', 'non-ironic']);
plt.show()

# Loading pre-trained image model
img_model = load_model('VGG16_img_model.h5')

# Compiling models
txt_model.compile(loss='binary_crossentropy', optimizer=adam, metrics = ["accuracy"])
img_model.compile(loss='binary_crossentropy', optimizer=adam, metrics = ["accuracy"])

# Concatenating the output of 2 classifiers
con_layer = keras.layers.concatenate([txt_model.output, img_model.output])
out = Dense(1,activation='sigmoid')(con_layer)

# Defining model input and output
com_model = Model(inputs = [img_model.input, txt_model.input], outputs=out)

# compiling the combined model
com_model.compile(loss='binary_crossentropy', optimizer=adam, metrics = ["accuracy"])

# Plot model
plot_model(com_model, to_file='Two_LSTM_Inception_mul_model.png', show_shapes=True, show_layer_names=True)

# Training model
com_model.fit_generator(img_txt_gen_train, epochs=7, validation_steps = 149, steps_per_epoch=2, validation_data=img_txt_gen_val, shuffle=False, class_weight=class_weight)

# saving combined model
com_model.save("Two_LSTM_Inception_mul_model.h5")

# Plotting training and validation loss for combined model
loss_values = com_model.history.history['loss']
val_loss_values = com_model.history.history['val_loss']
epochs = range(1, num_epochs + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predicting true labels using combined classifier
y_pred_com = (com_model.predict_generator(img_txt_gen_test,steps = 149))
y_pred_com = np.round(list(itertools.chain(*y_pred_com)))

# Confusion matrix
labels = [1,0]
cm = confusion_matrix(y_true, y_pred_com, labels=labels)
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax,cmap='Greens'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['offensive', 'non-offensive']); ax.yaxis.set_ticklabels(['offensive', 'non-offensive']);
# plt.figure(figsize=(5,6))

# Plotting model training accuracies
plt.plot(com_model.history.epoch, com_model.history.history['acc'])
plt.plot(txt_model.history.epoch, txt_model.history.history['acc'])
plt.gca().legend(('meme model acc', 'text model acc'))
plt.xlabel('epoch')
plt.ylabel('training accuracy')
plt.show()

# Plotting model validation accuracies
plt.plot(com_model.history.epoch, com_model.history.history['val_acc'])
plt.plot(txt_model.history.epoch, txt_model.history.history['val_acc'])
plt.gca().legend(('meme model validation acc', 'text model validation acc'))
plt.xlabel('epoch')
plt.ylabel('validaion accuracy')
plt.show()

# Loss and accuracy for combined model
com_model.evaluate_generator(img_txt_gen_test, steps=149)

# Loss and accuracy for text model
txt_model.evaluate_generator(txt_gen_test, steps=149)

# for txt
precision_recall_fscore_support(y_true, y_pred_txt, beta=1.0, labels=None, pos_label=1, average='binary')

# com model
precision_recall_fscore_support(y_true, y_pred_com, beta=1.0, labels=None, pos_label=1, average='binary')