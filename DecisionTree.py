# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import numpy as np
#import pandas
# Enter You Name Here
myname = "Saurav-vara-prasad-Channuri-" # or "Doe-Jane-"

# Implement your decision tree below


def entropy(dataset):

    count_one = 0
    count_zero = 0
   
    for x in np.nditer(dataset[1]):
       # print x
        if(x == '1'):
            count_one = count_one + 1
        else:
            count_zero = count_zero + 1

    total = count_one + count_zero

    entrpy_1 = 0
    entrpy_0 = 0

    if count_one != 0:
        entrpy_1 = -1*(float(count_one)/float(total)*np.log2(float(count_one)/float(total)))
    if count_zero != 0:
        entrpy_0 = -1*(float(count_zero)/float(total)*np.log2(float(count_zero)/float(total)))  

    entrpy =   entrpy_1 + entrpy_0
    return entrpy   


def information_gain(total_dataset, attribute, threshold):
    l_child = np.zeros(12)
    r_child = np.zeros(12)

    # print "len = ",len(total_dataset.T[0])
    # print "total_dataset = ", total_dataset
    for i in range(0,len(total_dataset.T[0])):                       # total_dataset  - columns are attributes    

        if(total_dataset[i][attribute] > threshold): 
            r_child = np.vstack((r_child,total_dataset[i][0:12]))
        else:
            l_child = np.vstack((l_child,total_dataset[i][0:12]))
        
    #print "r_child = ++  ", r_child[attribute]

    l_child_size = len(l_child)
    r_child_size = len(r_child)

    l_child_entropy = 0
    r_child_entropy = 0

#____________________________________________________________________________
   # print "l_child size = ", len(l_child) -1
   # print "r_child size = ", len(r_child) -1

    # print "r_child = ", r_child
#____________________________________________________________________________


    if len(r_child)!=12:
        r_child = np.delete(r_child, (0), axis=0)                        # r_child   -   columns are attributes                                   
        
        # print "r_child = ", r_child
        r_child_dataset = r_child.T[attribute]
        r_child_dataset = np.vstack((r_child_dataset, r_child.T[11]))
        r_child_dataset = r_child_dataset.T

        r_child_entropy = entropy(r_child_dataset.T)
        #r_child_size = len(r_child_dataset.T)


    if len(l_child)!=12:
        l_child = np.delete(l_child, (0), axis=0)                        # l_child   -   columns are attributes
    
        l_child_dataset = l_child.T[attribute]
        l_child_dataset = np.vstack((l_child_dataset, l_child.T[11]))
        l_child_dataset = l_child_dataset.T    

        l_child_entropy = entropy(l_child_dataset.T)
        #l_child_size = len(l_child_dataset.T)

    node_dataset = total_dataset.T[attribute]
    node_dataset = np.vstack((node_dataset,total_dataset.T[11]))
    parent_entropy = entropy(node_dataset)

    parent_size = l_child_size + r_child_size

    right_weighted_sum =  (float(r_child_size)/float(parent_size))*r_child_entropy
    left_weighted_sum = (float(l_child_size)/float(parent_size))*l_child_entropy
#___________________________________________________________________________________________
    # print "parent entropy = ", parent_entropy
    # print "l_child_size = ", l_child_size
    # print "r_child_size = ", r_child_size
    
    # print "l_child entropy = ", l_child_entropy
    # print "r_child entropy = ", r_child_entropy
    # print "fractions " , (float(l_child_size)/float(parent_size)), "  right =" ,(float(r_child_size)/float(parent_size))
    # print "right weighted sum = ", right_weighted_sum
    # print "left_weighted_sum = ", left_weighted_sum 
#___________________________________________________________________________________________

    child_entropy = right_weighted_sum + left_weighted_sum
  #  print "child_entropy = ", child_entropy
    knowledge_gained = parent_entropy - child_entropy

   # print "knowledge gained = ", knowledge_gained
    return knowledge_gained



class DecisionTree():
    tree = {}
    
    def learn(self, training_set):

        max_knowledge_gained = 0
        best_attribute = 0
        threshold_element = 0

        for attribute in range(0, len(training_set[0])-1):
            threshold_array = np.unique(training_set.T[attribute])
            print attribute
            for element in range(0,len(threshold_array)):
        
                #attribute = 0
                threshold = threshold_array[element]
                knowledge_gained = information_gain(training_set, attribute, threshold)
                 
            
                if knowledge_gained > max_knowledge_gained:
            
                    max_knowledge_gained = knowledge_gained
                    threshold_element = threshold 
                    best_attribute = attribute 

        print "threshold_element = ", threshold_element
        print "max max_knowledge_gained = ", max_knowledge_gained
        print "best attribute = ", best_attribute

            




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
