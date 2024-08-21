import os
import tensorflow.keras as keras
import random
import pickle
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = keras.Sequential()

lemmatizer = WordNetLemmatizer()

intents = json.load(open('intents.json'))

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
