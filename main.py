import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random

import json
with open('intents.json') as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

                                                    #stemming the words -> basically taking the words to their root

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)          #tokenize just gets us all the words from the patterns -> returns a list
        words.extend(wrds)                          #adding the words from pattern or rather extending wrds list into empty words list. 
        docs_x.append(wrds)                         #adding the words from pattern to doc.x
        docs_y.append(intent["tag"])                #adding the words form tag to doc.y
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])                #adding the tag uniquely to label list.

words = [stemmer.stem(w.lower()) for w in words if w != "?"]    #adding stemmed words to words list
words = sorted(list(set(words)))                    #set removes all the duplicates, we convert it back to list and then its sorted.

labels = sorted(labels)                             #setting the labels as well.

training = []
output = []

out_empty = [0 for _ in range(len(labels))]         #Adding zeros for each tags.
print(out_empty)

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)            #we need to convert them to numpy arrays, because we need them for usign tflearn
output = numpy.array(output)                #we need to convert them to numpy arrays, because we need them for usign tflearn
