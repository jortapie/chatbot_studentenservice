import nltk
# nltk.download()
import json
# import pickle

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    # print(intents['intents'])
    # TODO: Corregir repeticion de los intent.
    for pattern in intent['patterns']:
        # print(intent['patterns'])
        # tokenize here
        w = nltk.word_tokenize(pattern)
        print('Token is: {}'.format(w))
        words.extend(w)
        documents.append((w, intent['tag']))
        # add the tag to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

    # Final lists
    print('Words list is: {}'.format(words))
    print('Docs are: {}'.format(documents))
    print('Classes are: {}'.format(classes))

print('Eeeeexito')