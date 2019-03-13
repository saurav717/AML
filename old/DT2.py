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

    entropy_one = 0
    entropy_zero = 0

    if(count_one!=0):
        entropy_one = -(float(count_one)/float(total_data_size))*np.log2(float(count_one)/float(total_data_size))  
    if(count_zero!=0):
        entropy_zero = -(float(count_zero)/float(total_data_size))*np.log2(float(count_zero)/float(total_data_size))
    entrpy = entropy_zero + entropy_one  

    return entrpy 
   
def entropy(dataset, output_vector, threshold):


    # threshold = np.array(threshold).astype(float)
    # threshold = np.delete(threshold,11,0)
        
    #print "dataset = ", len(dataset)

    zeros_and_ones = 1*dataset
    total_data_size = len(dataset.T[0])

    total_ones = np.sum(output_vector, axis = 0)
    #print "total_ones = ", total_ones
    total_zeros = len(dataset) - total_ones

    child_count = zeros_and_ones.sum(axis = 0)
    #child_count = np.delete(child_count,11 ,0)

    #print zeros_and_ones                              # zeros and ones === rows * columns
    
    zeros_and_ones = zeros_and_ones.T
    # output_vector = np.asarray(output_vector)

    ones = output_vector*zeros_and_ones
    count_first_child_ones = np.sum(ones.T[:],axis = 0)


    #count_first_child_ones = np.delete(count_first_child_ones,11,0)
    count_first_child_zeros = child_count - count_first_child_ones
    
    count_second_child_ones = total_ones - count_first_child_ones
    count_second_child_zeros = total_zeros - count_first_child_zeros

    second_child_count = count_second_child_ones + count_second_child_zeros

    # count_second_child = len(dataset) - child_count 
    # # count_second_child_ones = 

    
    # print "child_count = ", child_count
    # print "count_child_zeros = ", count_first_child_zeros
    # print "count_child_ones = ", count_first_child_ones


    ones_entropy = -(count_first_child_ones/child_count) * np.log2(count_first_child_ones/child_count)
    zeros_entropy = -(count_first_child_zeros/child_count)*np.log2(count_first_child_zeros/child_count)

    second_ones_entropy = -(count_second_child_ones/second_child_count) * np.log2(count_second_child_ones/second_child_count)
    second_zeros_entropy = -(count_second_child_zeros/second_child_count) * np.log2(count_second_child_zeros/second_child_count)
 
    
    first_ones_entropy = np.nan_to_num(ones_entropy)
    first_zeros_entropy = np.nan_to_num(zeros_entropy)

    second_ones_entropy = np.nan_to_num(second_ones_entropy)
    second_zeros_entropy = np.nan_to_num(second_zeros_entropy)

    entrpy_one = first_ones_entropy + first_zeros_entropy
    entrpy_second = second_ones_entropy + second_zeros_entropy
    weight = ((child_count)/float(len(dataset)))
    weight_second = ((second_child_count)/float(len(dataset)))

    weighted_entropy = weight*entrpy_one  + weight_second*entrpy_second
    weighted_entropy = [weighted_entropy, entrpy_one, entrpy_second]
    # print "weighted entropy = ", np.array(weight).astype(float)* np.array(weighted_entropy).astype(float)
    # print "entropy = ", entrpy
    return weighted_entropy

def information_gain(training_set, output_vector):

    lent = len(training_set[0])
    print "lent = ", lent
    information_gain = np.zeros(lent-1)
    thresholds = np.zeros(lent)
    training_set.T[lent-1] = output_vector

    parent_entropy_val = parent_entropy(training_set)

    #training_set = np.array(training_set).astype(float)
    # output_vector = training_set.T[11]
    best_threshold = np.zeros(lent-1)
    best_information_gain = np.zeros(lent-1)

    icount = 0
    start = timeit.default_timer()
    for element in range(0,len(training_set)):
        # print "\n\n\t\t\t\t\t\t\t\t\t\t\t*****************************************************************************\t*****iterator******\t\t" , icount
        icount+=1
        threshold_array_index = element    
        threshold = training_set[threshold_array_index]

        dataset = training_set > threshold 
        entropy_children = entropy(dataset,output_vector, threshold)[0]

        threshold = np.delete(threshold,lent-1,0)
        # print "left_child_entropy = ", left_child_entropy
        # print "right_child_entropy = ", right_child_entropy

        information_gained = parent_entropy_val - (entropy_children)

        information_gained = np.array(information_gained).astype(float)
        information_gained = np.delete(information_gained,lent-1,0)

        bool_info_gain = ( np.array(best_information_gain).astype(float) < np.array(information_gained).astype(float))
        # print "bool = ", bool_info_gain

        best_information_gain = best_information_gain * ~bool_info_gain + information_gained * bool_info_gain

        best_threshold = best_threshold * ~bool_info_gain 
        best_threshold = best_threshold +  threshold * bool_info_gain
        
       
        # print "bth : " , best_threshold
        threshold = best_threshold
        information_gained = best_information_gain

    print "best threshold = ", best_threshold
    print "best information gain = ", best_information_gain
    stop = timeit.default_timer()

    print 
    print
    print
    print 
    print "time taken = ", stop - start 

    best_attribute = np.argmax(best_information_gain)
    node_threshold = best_threshold[best_attribute]
    best_information_value = best_information_gain[best_attribute]
    print "best attribute = ", best_attribute
    print "node threshold = ", node_threshold

    values = [best_attribute , node_threshold, best_information_value, best_threshold]
    return values


class node():
    def __init__(self, data, threshold, attribute_index, completed_attributes, knowledge_gained, filled ):
        self.data = data
        self.filled = 0
        self.threshold = threshold
        self.attribute_index = attribute_index
        self.completed_attributes = completed_attributes
        self.knowledge_gained = knowledge_gained
        self.left = None
        self.right = None 

    # def left_insert(self, data, threshold, attribute_index, completed_attributes, knowledge_gained):
    #     if self.

    def print_node(root):
        if root is not None:
            print_node(root.left)
            print "node threshold = ",root," node attribute = ", node.attribute_index

      
class DecisionTree():
    tree = {}
    
    def learn(self, training_set, completed_attributes, root, output_vector):
        print "give dummy input"
        deletethis = input()



        # output_vector = training_set.T[11]
        node_values = information_gain(training_set, output_vector)
        
        print "training set = ", training_set

        threshold_array = node_values[3]
        threshold_array = np.append(threshold_array, 0)
        training_set.T[11] = output_vector
       
        threshold = node_values[1]
        attribute = node_values[0]
        completed_attributes[attribute]=1
        best_information_gain = node_values[2] 

        print "threshold = ", threshold, "attribute = ", attribute, " best information gain = ", best_information_gain


        left_child = (1*(training_set.T[attribute] <= threshold_array[attribute]))
        right_child = (1*(training_set.T[attribute] > threshold_array[attribute]))
        
        left_child = np.repeat([left_child], 12, axis = 0).T
        right_child = np.repeat([right_child], 12, axis = 0).T

        # print "left_child = ", (np.count_nonzero(right_child.T[0]))
        print "right_child size= ", np.count_nonzero(right_child.T[2])
        print "left_child size= ", np.count_nonzero(left_child.T[2])

        left_child_dataset = training_set*left_child
        right_child_dataset = training_set*right_child

        print "left_child_dataset"
        print left_child_dataset
        print "right child datatset"
        print right_child_dataset

        print "left child dataset = ", np.count_nonzero(left_child_dataset.T[0])
        print "right child dataset = ", np.count_nonzero(right_child_dataset.T[0])
        


        entropies = entropy(training_set, output_vector, threshold )[1]
        right_child_entropy = entropies[2]
        left_child_entropy = entropies[1]

        # print "left child entropy = ", left_child_entropy
        # print "right child entropy = ", right_child_entropy
        # print"left_child_dataset = ", left_child_dataset
        # print "output vector = ", output_vector
        print
        print


        if root == None:
            print "root is null there fore creating root"
            root = node(training_set, threshold, attribute, completed_attributes, best_information_gain, 1)


        print "root attribute = ",root.attribute_index

        # if best_information_gain == '0.0':
        #     return output_vector[attribute]

        #tree = DecisionTree()
        if (np.float(left_child_entropy) != '0' or np.float(best_information_gain) != '0' or np.count_nonzero(completed_attributes) != '0'):
            print 
            print
            print "this works"
            root_left = self.learn(left_child_dataset, completed_attributes, None, output_vector)
            root.left = root_left
        if (right_child_entropy != '0' or best_information_gain != '0' or np.count_nonzero(completed_attributes) != '0'): 
            print
            print
            print "this works 2"   
            root_right = self.learn(right_child_dataset, completed_attributes, None, output_vector)
            root.right = root_right

        return root 

    # implement this function
    def classify(self, test_instance):
        # print "root_threshold $$$$$$$$$$$$$$$$$$$$$$$$$$$$$ = ", root.threshold
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
    #root = node(None, None, None, None, None)
    # Construct a tree using training set
    completed_attributes = np.zeros(11)

    training_set = np.array(training_set).astype(float)
    root = tree.learn( np.array(training_set).astype(float),completed_attributes, None,training_set.T[11] )

    print "root_threshold ======= ", root.threshold
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
