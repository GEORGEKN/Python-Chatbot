import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open('intents.json') as file:
    data = json.load(file)

try: 
    with open("data.pickle", 'rb') as f :# rb-- read bytes#   save data as bytes
        words, labels, training, output = pickle.load(f)        

except:      

    words = [] # all the different words
    labels = []
    docs_x = [] # list of all the different patterns
    docs_y = []  

    for intent in data['intents']:
        for pattern in intent ['patterns']:
            wrds = nltk.word_tokenize(pattern) # returns a list with all the different words in it
            words.extend(wrds) # rather than looping and appending each of the words, .extend() add all the words in  
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    ### pre processing ####
    words = [stemmer.stem(w.lower())for w in words if w != '?']
    words = sorted(list(set(words))) ## set(words)- removes duplicate words

    labels = sorted(labels)

    ## we now have strings but neural newtworks only understands numbers
    ## Now create a bag of words( one- hot encoding) that represent all of the words in any given pattern  #https://analyticsindiamag.com/when-to-use-one-hot-encoding-in-deep-learning/
    ## and we are going to use that to train the model


    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []#  bag of word

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    # change training and output to arrays so as to feed them into the model

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", 'wb') as f :
        pickle.dump((words, labels, training, output),f)

# building the model

tensorflow.compat.v1.reset_default_graph() # get rid of all previous settings

net = tflearn.input_data(shape = [None, len(training[0])]) # defines the input shape we are expecting for our model
net = tflearn.fully_connected(net,8) # hidden layer with eight neurons
net = tflearn.fully_connected(net,8) # hidden layer with eight neurons
net = tflearn.fully_connected(net,len(output[0]),activation = 'softmax') #output layer... (softmax)- gives probabilities for each output
net = tflearn.regression(net)

# training the model

model = tflearn.DNN(net) ## DNN - type of neural network

# Fitting the model
# try and except is being used to prevent training the model when the model already exists
try:
    model.load('model.tflearn')
except:    
    model.fit(training, output, n_epoch =1000, batch_size = 8, show_metric = True)
    model.save('model.tflearn') 

### Making predictions ###

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))] # creates a blank bag of words list and then change the elements in here to represent if a word exists or if it doesn't 

    s_words = nltk.word_tokenize(s) ## word_tokenize is a function in Python that splits a given sentence into words
    s_words = [stemmer.stem(word.lower()) for word in s_words]  ## Stemmers remove morphological affixes from words, leaving only the word stem

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i]= 1

    return numpy.array(bag)

## function that asks the user for for some sentence
## and then ngive out a reponse 
def chat():
    print('Start talking with the bot (type quit to stop)!')
    while True:
        inp = input('You:')
        if inp.lower() == 'quit':
            break

        results = model.predict([bag_of_words(inp,words)]) [0] 
        results_index = numpy.argmax(results) # Gives the index of the greatest value of the list** where the value is the classification from the model 
        tag = labels[results_index] 

        if results[results_index] > 0.7:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print (random.choice(responses)) 
        else:
            print('I did not understand, please try again.')        
        

chat()                      
