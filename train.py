# Simple Bag-of-Words classifier for sentiment classification of restaurant reviews with Pytorch
# Data from Xiang Zhang: 
# https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?resourcekey=0-TLwzfR2O-D2aPitmn5o9VQ
# (yelp_review_polarity_csv)

import torch
import torch.optim as optim
from collections import Counter
import csv

from bow import make_word_dictionary, make_label_dictionary, make_bow_vector, make_label_vector
from model import BoWClassifier

torch.manual_seed(1) # reproduceability

def read_data_from_file(path, num_rows=None):
    data = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, quotechar='"', doublequote=True, escapechar='\\')
        for i, row in enumerate(reader):
            if num_rows is not None and i >= num_rows:
                break
            data.append((row[1].split(), row[0]))  # [1] = text, [0] = label
    return data

# Get data - use only a part of it to speed up training/testing
training_data = read_data_from_file('data/train.csv', 10000)
test_data = read_data_from_file('data/test.csv', 1000)

# get target class distribution in train and test, should be ~50%
print(Counter(label for txt, label in training_data)) 
print(Counter(label for txt, label in test_data))

# Get dictionaries
word_dictionary = make_word_dictionary(training_data, 10)   # UNKs are not handled here, see "make_word_dictionary()"
label_dictionary = make_label_dictionary(training_data)

print(len(word_dictionary))

# Initialize classifier
model = BoWClassifier(len(word_dictionary), len(label_dictionary))

# Define loss and optimizer
loss_function = torch.nn.NLLLoss() # model returns log probs

optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training
number_of_epochs = 5
for epoch in range(number_of_epochs):

    print(epoch)

    for instance, label in training_data:

        model.zero_grad()

        bow = make_bow_vector(instance, word_dictionary)  # UNKs are handled in "make_bow_vector()"
        target = make_label_vector(label, label_dictionary)

        log_probs = model.forward(bow)

        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

# Evaluattion
def evaluate_final_model(model, test_data, word_dictionary, label_dictionary) -> float:

    correct_predictions: int = 0

    with torch.no_grad():

        # go through all test data points
        for instance, label in test_data:

            # get the vector for the data point
            bow = make_bow_vector(instance, word_dictionary)

            # send through model to get a prediction
            log_probs = model.forward(bow)

            label_idx = log_probs.argmax().item() # get index of max value in result vector = class with max. predicted probability
            predicted_label = list(label_dictionary.keys())[label_idx] # get label for predicted class

            if(predicted_label == label):
                correct_predictions += 1

        return correct_predictions/len(test_data)


accuracy = evaluate_final_model(model, test_data, word_dictionary, label_dictionary)
print(accuracy) #  ~ 0.832
