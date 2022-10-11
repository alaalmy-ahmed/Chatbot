import nltk  # Natural language Processing Toolkit
nltk.download('punkt')  # put an integer to a specific word
from nltk.stem.lancaster import LancasterStemmer  # Returns the words like plays, playing and played to the original word play.
stemmer = LancasterStemmer()  

import numpy  # Linear Algebra Library.
import tensorflow as tf  # Tensorflow to train our model.
import tflearn  #
import random  # to pick a random answer to the user's question.
import json  # to read the training data file.
import pickle  # to save the data after the preprocessing step.

from time import sleep  # to delay the answering process to simulate the writing process.

# Reading the training data.
with open("intents.json") as file:
    data = json.load(file)

# try and except, if this is the first time to run the model, then the model will be created and trained, else the model will be loaded.
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)  # load the training data.
except:
    words = []  # All the words in the training data.
    labels = []  # All the tags in the training data.
    docs_x = []   # words without tokenization.
    docs_y = []  # relations between the words and tags.
    for intent in data ["intents"]:  # Go to the intents section
        for pattern in intent["patterns"]:  # Select the pattern
            wrds = nltk.word_tokenize(pattern)   # Tokenize the words.
            words.extend(wrds)  # adding the words to words list.
            docs_x.append(wrds)  # add the words to the  list of words with tokenizations.
            docs_y.append(intent["tag"])  # adding the relation between the word and the tag.

        if intent["tag"] not in labels:
            labels.append(intent["tag"])  # Adding new tags tot the tags section.

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]  # stemming the words and removing the question marks.
    words = sorted(list(set(words)))  # sort the words to avoid duplicates.

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]  # Zeroes list with the size of the labels list.

    for x, doc in enumerate(docs_x):  # iterate over the training words.
        bag = []  # bagging the tokenized words.

        wrds = [stemmer.stem(w) for w in doc]  # stemming words in the original list of words. 

        for w in words:
            if w in wrds:
                bag.append(1)  # if the word exists in the our data words list, append 1 to the bag list. otherwise, append 0 .
            else:
                bag.append(0)


        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)  # converting the training data into a numpy arrays to train the model on them.
    output = numpy.array(output)   # converting the output data into a numpy arrays to train the model on them.

    # Saving the Shaped data because obviously we will not shape it everytime we run the script.
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tf.compat.v1.reset_default_graph()  # remove any default graph in the Cash.

# Creating the neural network.
net = tflearn.input_data(shape=[None, len(training[0])])  # input layer with the input shape.

# fully Hidden connected layers.
net = tflearn.fully_connected(net, 64)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 8)

# Output layer.
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#  Same thing we will not train the model everytime so we will save it
#  after the training and if it exists we will load it otherwise train it from  scratch
try:
    model.load("model.tflearn")
except:
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=16, show_metric=True)
    model.save("model.tflearn")

# Turning the user's input into a bag of integers so the model can understand it.
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


# Interactive function to interact with the user.
def chat():
    print("Hi, How can i help you ?")  # the Greeting sentence always.
    while True:
        inp = input("You: ")  # take a query from the user.
        if inp.lower() == "quit":  # if you want to exit the program type "quit".
            break

        results = model.predict([bag_of_words(inp, words)])[0]  # predict the sentence.
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        # if the probability of the predicted sentence is higher than 60% then pick a respons otherwise print "I don't understand"
        if results[results_index] > 0.5:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            sleep(3)
            Bot = random.choice(responses)
            print(Bot)
        else:
            print("I don't understand!")
chat()