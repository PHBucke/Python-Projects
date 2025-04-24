import pandas as pd
import numpy as np
import keras
import tensorflow
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense


data = pd.read_csv("Train.pdf", sep=';')
data.columns = ["Text", "Emotions"]
print(data.head())

texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
max_lenght = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_lenght)


label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

one_hot_labels = keras.utils.to_categorical(labels)

#Crindo os splits de dados e separando-os entre treino e teste do modelo
xtrain, xtest, ytrain, ytest = train_test_split(padded_sequences, one_hot_labels, test_size=0.2)

#Definição do modelo com seus parâmetros
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                    output_dim=128, input_length=max_lenght))

model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=len(one_hot_labels[0]), activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_data=(xtest, ytest))

input_text = "she is very mad with her relationship"

input_sequence = tokenizer.texts_to_sequences([input_text])
padded_input_sequence = pad_sequences(input_sequence, maxlen=max_lenght)
prediction = model.predict(padded_input_sequence)
prediction_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
print("The emotion in this phrase is: ", prediction_label)
