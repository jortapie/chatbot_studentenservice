import nltk
nltk.download()
import json
# import pickle

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize here
        w = nltk.word_tokenize(pattern)
        print('Token is: '.format(w))
        words.extend(w)
        documents.append((w, intent['tag']))