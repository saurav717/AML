# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import numpy as np
import operator
import timeit
#import pandas
# Enter You Name Here
myname = "Saurav-vara-prasad-Channuri-" # or "Doe-Jane-"

# Implement your decision tree below


def parent_entropy(dataset):
    
    total_data_size = len(dataset.T[0])
    
    outputs = map(int, dataset.T[11])
    count_one = np.count_nonzero(outputs)
    count_zero = total_data_size - count_one

    if(count_one!=0):
        entropy_one = -(float(count_one)/float(total_data_size))*np.log2(float(count_one)/float(total_data_size))  
    if(count_zero!=0):
        entropy_zero = -(float(count_zero)/float(total_data_size))*np.log2(float(count_zero)/float(total_data_size))
    entrpy = entropy_zero + entropy_one  

    return entrpy 
   
def entropy(dataset, output_vector):

    zeros_and_ones = 1*dataset

    total_data_size = len(dataset.T[0])

    child_count = zeros_and_ones.sum(axis = 0)
    child_count = np.delete(child_count,11 ,0)

    #print zeros_and_ones                              # zeros and ones === rows * columns
    
    zeros_and_ones = zeros_and_ones.T
    output_vector = np.asarray(output_vector)

    ones = np.array(output_vector).astype(float)*np.array(zeros_and_ones).astype(float)
    count_first_child_ones = np.sum(ones.T[:],axis = 0)


    count_first_child_ones = np.delete(count_first_child_ones,11,0)
    count_first_child_zeros = child_count - count_first_child_ones
    
    # count_second_child = len(dataset) - child_count 
    # # count_second_child_ones = 

    
    # print "child_count = ", child_count
    # print "count_child_zeros = ", count_first_child_zeros
    # print "count_child_ones = ", count_first_child_ones

    # # child_count = np.array(child_count,dtype = np.float)
    # count_child_ones = np.array(count_child_ones,dtype = np.float)
    # count_child_zeros = np.array(count_child_zeros,dtype = np.float)

    # print "divison = ", np.log2(count_child_zeros/child_count)

    #zeros_child_ones = 
    # iszeroOnes = count_child_ones == np.zeros(len(count_child_ones))
    # iszeroZeros = count_child_zeros == np.zeros(len(count_child_zeros))
    # print iszeroOnes 
    # print iszeroZeros

    ones_entropy = -(count_first_child_ones/child_count) * np.log2(count_first_child_ones/child_count)
    zeros_entropy = -(count_first_child_zeros/child_count)*np.log2(count_first_child_zeros/child_count)
 
    
    first_ones_entropy = np.nan_to_num(ones_entropy)
    first_zeros_entropy = np.nan_to_num(zeros_entropy)

    entrpy = first_ones_entropy + first_zeros_entropy
    weight = ((child_count)/float(len(dataset)))
    weighted_entropy = weight*entrpy
    # print "weighted entropy = ", np.array(weight).astype(float)* np.array(weighted_entropy).astype(float)
    # print "entropy = ", entrpy
    return weighted_entropy



def information_gain(training_set):

    information_gain = np.zeros(11)
    thresholds = np.zeros(11)

    parent_entropy_val = parent_entropy(training_set)

    output_vector = training_set.T[11]
    best_threshold = np.zeros(11)
    best_information_gain = np.zeros(11)

    icount = 0
    start = timeit.default_timer()
    for element in range(0,len(training_set)):
        print "\n\n\t\t\t\t\t\t\t\t\t\t\t*****************************************************************************\t*****iterator******\t\t" , icount
        icount+=1
        threshold_array_index = element    
        threshold = training_set[threshold_array_index]

        right_children_matrix = training_set > threshold
        left_children_matrix = training_set <= threshold 

        right_child_count = right_children_matrix.sum(axis = 0)
        right_child_count = np.delete(right_child_count,11 ,0)

        left_child_count = left_children_matrix.sum(axis = 0)
        left_child_count = np.delete(left_child_count,11 ,0)


        # print "left child count = ", left_child_count
        # print "right child count = ", right_child_count
        # print
        # print
        # print "parent entropy = ", parent_entropy_val

        # print "training_set = ", training_set
        # print "right_children_mat"
        threshold = np.array(threshold).astype(float)
        threshold = np.delete(threshold,11,0)
        weighted_entropy_right = entropy(right_children_matrix, output_vector)
        weighted_entropy_left =  entropy(left_children_matrix, output_vector)

        # print "left_child_entropy = ", left_child_entropy
        # print "right_child_entropy = ", right_child_entropy

        information_gained = parent_entropy_val - (weighted_entropy_left + weighted_entropy_right)

        bool_info_gain = ( best_information_gain < information_gained)
        # print "bool = ", bool_info_gain

        # print "information gained = ", information_gained
        # print "old information    = ", best_information_gain
        # print "*********************************************************************************************************"

        best_information_gain = best_information_gain * ~bool_info_gain + information_gained * bool_info_gain

        best_threshold = best_threshold * ~bool_info_gain 
        best_threshold = best_threshold +  threshold * bool_info_gain
        
        # best_threshold = np.array(best_threshold).astype(float)* ~np.array(bool_info_gain).astype(float) + np.array(threshold).astype(float)* np.array(bool_info_gain).astype(float)
        # best_threshold = np.array(best_threshold).astype(float)* ~np.array(bool_info_gain).astype(float) + np.array(threshold).astype(float)* np.array(bool_info_gain).astype(float)
       
        print "bth : " , best_threshold
        # threshold = best_threshold
        # information_gained = best_information_gain

    print "best threshold = ", best_threshold
    print "best information gain = ", best_information_gain
    stop = timeit.default_timer()
    print "time taken = ", stop - start 
        # print "right child matrix = ", right_children_matrix
        # print "left child matrix = ", left_children_matrix

        # # left_children_matrix = left_children_matrix

        # right_child_size = (1*right_children_matrix).sum(axis = 0)
        # left_child_size = (1*left_children_matrix).sum(axis = 0)

        # left_child_size = np.delete(left_child_size,11 ,0)
        # right_child_size = np.delete(right_child_size,11 ,0)
       
        # left_child_size = map(float, left_child_size)
        # right_child_size = map(float, right_child_size)

        # # # total_data_size = np.sum(left_child_size, right_child_size)
        # # total_data_size = map(operator.add, left_child_size, right_child_size)

        # # print "right child size = ",len(right_child_entropy)
        # # print "left child size = ",len(left_child_size)
        # # print "left child entropy = ", left_child_entropy
        # # print
        # # print "right child entropy =  ", right_child_entropy
        # # print
        # left_child_fraction = np.divide((left_child_size),(total_data_size))
        # right_child_fraction = np.divide((right_child_size),(total_data_size))

        # # print "left child fraction size = ", len(left_child_fraction)

        # # print "left_child_ fraction = ", left_child_fraction
        # # print
        # # print"right child fraction = ", right_child_fraction
        # # print
        # # print "parent entropy values = ", parent_entropy_val

        # weighted_entropy_left =  left_child_fraction * left_child_entropy
        # weighted_entropy_right = right_child_fraction * right_child_entropy

        # # print
        # # print "weighted entropy left = ", weighted_entropy_left
        # # print
        # # print "weighted entropy right = ", weighted_entropy_right
        # # print

        # information_gained =  parent_entropy_val - ((weighted_entropy_right + weighted_entropy_left) )
        # # print "information_gain = ", information_gained
    # print "weighted entropy right  = ", len(right_children_matrix) 

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
