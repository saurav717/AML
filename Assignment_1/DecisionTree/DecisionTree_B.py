# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import numpy as np
import operator
import timeit

# Enter You Name Here
myname = "Saurav-vara-prasad-Channuri-" # or "Doe-Jane-"

# Implement your decision tree below


def parent_entropy(dataset):

    total_data_size = len(dataset.T[0])

    attribute_size = len(dataset[0])

    outputs = map(int, dataset.T[attribute_size-1])
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

    zeros_and_ones = 1*dataset
    total_data_size = len(dataset.T[0])
    attribute_size = len(dataset[0])
 
    entropy_one = 0
    entropy_zero = 0
 
    total_ones = np.sum(output_vector, axis = 0)
    total_zeros = len(dataset) - total_ones

    child_count = zeros_and_ones.sum(axis = 0)
 
    zeros_and_ones = zeros_and_ones.T
 
    ones = output_vector*zeros_and_ones
    count_first_child_ones = np.sum(ones.T[:],axis = 0)


    count_first_child_zeros = child_count - count_first_child_ones

    count_second_child_ones = total_ones - count_first_child_ones
    count_second_child_zeros = total_zeros - count_first_child_zeros

    second_child_count = count_second_child_ones + count_second_child_zeros

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
  
    return weighted_entropy

def information_gain(training_set, output_vector):

    lent = len(training_set[0])
    information_gain = np.zeros(lent-1)
    thresholds = np.zeros(lent)

    parent_entropy_val = parent_entropy(training_set)

    output_vector = training_set.T[lent-1]
    best_threshold = np.zeros(lent-1)
    best_information_gain = np.zeros(lent-1)

    icount = 0
    start = timeit.default_timer()
    for element in range(0,len(training_set)):

        icount+=1
        threshold_array_index = element
        threshold = training_set[threshold_array_index]

        dataset = training_set > threshold
        output_vector = training_set.T[len(training_set[0])-1]
        entropy_children = entropy(dataset,output_vector, threshold)[0]

        threshold = np.delete(threshold,lent-1,0)

        information_gained = parent_entropy_val - (entropy_children)
        information_gained = np.array(information_gained).astype(float)
        information_gained = np.delete(information_gained,lent-1,0)

        bool_info_gain = ( np.array(best_information_gain).astype(float) < np.array(information_gained).astype(float))

        best_information_gain = best_information_gain * ~bool_info_gain + information_gained * bool_info_gain

        best_threshold = best_threshold * ~bool_info_gain
        best_threshold = best_threshold +  threshold * bool_info_gain


        threshold = best_threshold
        information_gained = best_information_gain

    stop = timeit.default_timer()


    best_attribute = np.argmax(best_information_gain)
    node_threshold = best_threshold[best_attribute]
    best_information_value = best_information_gain[best_attribute]
  
  
    values = [best_attribute , node_threshold, best_information_value, best_threshold]
    return values


def classification (test_instance, root, training_set):
    
    if(root.left == None and root.right != None):
        return classification(test_instance, root.right, training_set)
    
    elif(root.left != None and root.right == None):
        return classification(test_instance, root.left, training_set)
    
    elif(root.left == None and root.right == None):
        if root.ones > (root.total - root.ones):
            return 1

        else:
            return 0

    if test_instance[root.attribute_index] <= root.threshold:
        if root.left == None:
            if(root.ones > (root.total - root.ones)):
                return 1

            else:
                return 0

        else:
            return classification(test_instance, root.left, training_set)

    elif test_instance[root.attribute_index] > root.threshold:
        if(root.right == None):
            if(root.ones > (root.total - root.ones)):
                return 1

            else:
                return 0

        else:
            return classification(test_instance, root.right, training_set)



class node():
    def __init__(self, data, threshold, attribute_index, completed_attributes, knowledge_gained, filled, ones, total ):
        self.data = data
        self.filled = 0
        self.threshold = threshold
        self.attribute_index = attribute_index
        self.completed_attributes = completed_attributes
        self.knowledge_gained = knowledge_gained
        self.ones = ones
        self.total = total
        self.left = None
        self.right = None

   
    def print_node(root):
        if root is not None:
            print_node(root.left)



class DecisionTree():
    tree = {}

    def learn(self, training_set, completed_attributes, root, output_vector, completed_attributes_final):
      
        # print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        node_values = information_gain(training_set, output_vector)
        

        threshold_array = node_values[3]
        threshold_array = np.append(threshold_array, 0)

        threshold = node_values[1]
        attribute = node_values[0]
      
        real_attribute = completed_attributes_final[attribute]

        temp_attributes =np.linspace(0,len(completed_attributes)-1, len(completed_attributes))
        completed_attributes = np.delete(np.array(temp_attributes).astype(int), attribute, 0)
        completed_attributes_final = np.delete(np.array(temp_attributes).astype(int), attribute, 0)

        best_information_gain = node_values[2]  
      
        left_child = (1*(training_set.T[attribute] <= threshold_array[attribute]))
        right_child = (1*(training_set.T[attribute] > threshold_array[attribute]))

        left_child_indexes = np.argwhere(left_child > 0).T[0]
        right_child_indexes = np.argwhere(right_child > 0).T[0]


      
        training_set_temp = training_set[:,completed_attributes]


        left_child_dataset = training_set_temp[left_child_indexes,:]
        right_child_dataset = training_set_temp[right_child_indexes,:]
        attribute_size = len(training_set[0])
        output_vector_temp = training_set.T[attribute_size-1]
       
        entropies_temp = entropy(training_set, output_vector, threshold )
        entropies = entropies_temp[1]

        ones_sum = output_vector_temp.sum(axis = 0)
        total = len(output_vector_temp)

        root = node(training_set, threshold, real_attribute, completed_attributes, best_information_gain, 1, ones_sum, total)

       
        right_child_entropy = 0
        left_child_entropy = 0

        if(attribute_size >= 3):
            right_child_entropy = entropies[2]
            left_child_entropy = entropies[1]

     
        if (np.float(best_information_gain) != 0 and np.count_nonzero(completed_attributes) != 0 and attribute_size > 2 and len(left_child_dataset)!=0):
            # print
            output_vector = output_vector_temp[left_child_indexes]
            root_left = self.learn(left_child_dataset, completed_attributes, None, output_vector,completed_attributes_final)
            root.left = root_left
        elif(np.float(best_information_gain) != 0 and np.count_nonzero(completed_attributes) != 0 and attribute_size > 2 and len(right_child_dataset)!=0):
            # print
            output_vector = output_vector_temp[right_child_indexes]
            root_right = self.learn(right_child_dataset, completed_attributes, None, output_vector, completed_attributes_final)
            root.right = root_right

        return root

    # implement this function


    def classify(self, test_instance, root, training_set):

        result = classification(test_instance, root, training_set)
        return result



def run_decision_tree():

    # Load data set
    with open("wine-dataset.csv") as f:
        next(f, None)
        data = [tuple(line) for line in csv.reader(f, delimiter=",")]

    k_divisions = 10

    A = np.linspace(0,len(data), k_divisions+1)
    B = np.array(A).astype(int)

    data = np.array(data).astype(float)
    tris = np.asarray(data)

    # np.random.shuffle(data)

    X1 = data[B[0]:B[1]]
    X2 = data[B[1]:B[2]]
    X3 = data[B[2]:B[3]]
    X4 = data[B[3]:B[4]]
    X5 = data[B[4]:B[5]]
    X6 = data[B[5]:B[6]]
    X7 = data[B[6]:B[7]]
    X8 = data[B[7]:B[8]]
    X9 = data[B[8]:B[9]]
    X10_testing_data = data[B[9]:B[10]]


    # print "len = ",len(X10_testing_data)
    # print len(X2)
    # print len(X3)
    # print len(X4)
    # print len(X5)
    # print len(X6)
    # print len(X7)
    # print len(X8)
    # print len(X9)

    training_data_list = [X1, X2, X3, X4, X5, X6, X7, X8, X9, X10_testing_data]
    # training_set = np.roll(training_set, 1 , 0)
    # training_set = np.array(training_set).astype(float)

    sum_accuracy = 0
    for i in range (0,10):

        training_set = training_data_list[0].T
        for i in range (0,8):
            training_set = np.hstack((training_set, training_data_list[i+1].T))

        # print "Number of records: %d" % len(training_set.T)

        # You need to modify the following code for cross validation.
        K = 10
        tree = DecisionTree()
      
        completed_attributes = [0,1,2,3,4,5,6,7,8,9,10,11]

        root = tree.learn( training_set,completed_attributes, None,training_set.T[11], completed_attributes)
      
        # Classify the test set using the tree we just constructed
        results = []
        test_set = training_data_list[9]
        for instance in test_set:
             result = tree.classify( np.array(instance[:-1]).astype(float) ,root, training_set )
             results.append( float(result) == float(instance[11]) )

        accuracy = float(results.count(True))/float(len(results))
        
        print "accuracy: %.4f" % accuracy
        training_data_list = np.roll(training_data_list,1,0)
        sum_accuracy = sum_accuracy + accuracy
        print "sum accuracy = ", sum_accuracy


    final_accuracy = sum_accuracy
    print "final accuracy = ", final_accuracy/10
    # Writing results to a file (DO NOT CHANGE)
    f = open(myname+"result.txt", "w")
    f.write("accuracy: %.4f" % final_accuracy)
    f.close()

if __name__ == "__main__":
    run_decision_tree()
