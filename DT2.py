# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import numpy as np
#import pandas
# Enter You Name Here
myname = "Saurav-vara-prasad-Channuri-" # or "Doe-Jane-"

# Implement your decision tree below


def entropy(dataset):

   


def information_gain(training_set):

    information_gain = np.zeros(11)
    thresholds = np.zeros(11)

    for threshold_array_index in range(0,len(training_set)):

        new_entropies = entropy(dataset, training_set[threshold_array_index])




class DecisionTree():
    tree = {}
    
    def learn(self, training_set):
        information_gain(training_set)


    # implement this function
    def classify(self, test_instance):
        result = 0 # baseline: always classifies as 0
        return result



def run_decision_tree():

    # Load data set
    with open("wine-dataset.csv") as f:
        next(f, None)
        data = [tuple(line) for line in csv.reader(f, delimiter=",")]

    tris = np.asarray(data)
    
    X1 = data[0:500]
    X2 = data[500:1000]
    X3 = data[1000:1500]
    X4 = data[1500:2000]
    X5 = data[2000:2500]
    X6 = data[2500:3000]
    X7 = data[3000:3500]
    X8_validation = data[3500:4000]

    training_set = [X1, X2, X3, X4, X5, X6, X7, X8_validation]
    
    #print len(training_set)
    # print (tris)
    print "Number of records: %d" % len(data)

    #________________________________________________________________
    #
    # Split training/test sets
    #________________________________________________________________
    #
    #
    # You need to modify the following code for cross validation.
    K = 10
    training_set = [x for i, x in enumerate(data) if i % K != 9]
    test_set = [x for i, x in enumerate(data) if i % K == 9]
    #________________________________________________________________

    
    #________________________________________________________________
    #
    #  Training the tree
    #________________________________________________________________
    #
    tree = DecisionTree()
    # Construct a tree using training set
    tree.learn( np.asarray(training_set) )
    #
    #________________________________________________________________

    

    # Classify the test set using the tree we just constructed
    results = []
    for instance in test_set:
        result = tree.classify( instance[:-1] )
        results.append( result == instance[-1])

    # Accuracy
    accuracy = float(results.count(True))/float(len(results))
    print "accuracy: %.4f" % accuracy       
    

    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result.txt", "w")
    f.write("accuracy: %.4f" % accuracy)
    f.close()


if __name__ == "__main__":
    run_decision_tree()
