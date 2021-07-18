import pickle

import nltk
# nltk.download()
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import json

import random
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

words = []
classes = []
documents = []
# TODO: Buscar una librería que incluya palabras a ignorar
ignore_words = ['?', '!', '.']

# Lee un .json con la lista de intents
data_file = open('intents.json').read()
intents = json.loads(data_file)

# En esta sección identificamos los intents y los patters y los asociamos a documentos para el posterior entrenamiento

# Extrae cada componente del .json por intents
for intent in intents['intents']:
    # TODO: Corregir repetición de los intent.
    # Extrae cada componente de los intents por pattern (oraciones)
    for pattern in intent['patterns']:
        # Tokeniza cada pattern en cada una de las palabras de la oración
        w = nltk.word_tokenize(pattern)
        # print('Token is: {}'.format(w))
        # Añade todas las palabras tokenizada de un mismo pattern a una lista de palabras
        words.extend(w)
        # Cada palabra tokenizada por pattern (oración) se le asigna un tag
        documents.append((w, intent['tag']))  # (['hey', 'you'], 'greeting')
        # Añade los tag a la lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    # Final lists
    # print('Words list is: {}'.format(words))
    # print('Docs are: {}'.format(documents))
    # print('Classes are: {}'.format(classes))

# Lematiza las palabras y las guarda como un array
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# Convierte la lista en un set para eliminar duplicados y vuelve a convertir a una lista para enviarla a un pickle
# print(words)
words = list(set(words))
classes = list(set(classes))

# Añade las listas a un archivo pickle para poder ser reutilizados
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# En esta sección creamos el código para el entrenamiento del algoritmo
# Se pretende crear dos set: observaciones y lables
training = []
output_empty = [0] * len(classes)

# Extrae componente por componente de los documentos para evaluarlos
for doc in documents:
    bag = []
    pattern_words = doc[0]
    # TODO: Revisar si la variable es word o words
    # Extrae de cada palabra de cada oracion, las lematiza y los guarda como un array
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # print('Current Pattern Word: {}'.format(pattern_words))

    # Compara las palabras extraídas de words (en el bloque anterior) y las compara con las palabras extraídas
    # de este bloque en pattern_words.
    # Le agrega un 1 a la bolsa si coincide y un 0 si no coincide
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # print('Current bag {}'.format(bag))

    # Convertimos a un array y asignamos la clase del documento a un array Y (label)
    output_row = list(output_empty)
    # Del array de clases identificamos la ubicación del clase, tomamos el index
    # y se lo asignamos al array de output_row
    output_row[classes.index(doc[1])] = 1
    # print('Current output {}'.format(output_row))
    # Creación la matriz de aprendizaje
    training.append([bag, output_row])
    # print('Training: {}'.format(training))

# Para que el programa no se acostumbre a un patrón de data se shuffle la data
random.shuffle(training)
training = np.array(training, dtype=object)

# Separa los datos de entrenamiento entre las observaciones y los labels
train_x = list(training[:, 0])
train_y = list(training[:, 1])
# print('x: {}'.format(train_x))
# print('y: {}'.format(train_y))

# En esta sección preparamos la red neuronal para que el algoritmos aprenda los datos

# Ingreso el modelo de deeplearning Sequential
model = keras.Sequential()
# Ingreso del modelo de la primera capa como 128 neuronas, dimensiones de entrada y la función de activación
model.add(layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(layers.Dropout(0.5))
# Ingreso del modelo de la segunda capa como 128 neuronas, dimensiones de entrada y la función de activación
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
# Ingreso del modelo del output, dimensiones de salida y la función de salida
model.add(layers.Dense(len(train_y[0]), activation='softmax'))

# Función de optimización con un learning rate de 0.01
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Compilación del modelo basado en la accuracity
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Ingreso de datos dentro del modelo
mfit = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', mfit)
print('Eeeeexito')