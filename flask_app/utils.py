import os
import random
import pickle
import json
import numpy as np
import nltk
import tensorflow.keras as keras
from nltk.stem import WordNetLemmatizer

model = keras.load_model


def clean_up_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    ignoreLetters = ['?', '!', '.', ',']

    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words if word not in ignoreLetters]
    return sentence_words


def bag_of_words(sentence):
    words = pickle.load(open('model/words.pkl', 'rb'))
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)
