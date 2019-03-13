	# CS6510 HW 1 Code Skeleton
# Please use this outline to implement your decision tree. You can add any code around this.

import csv
import numpy as np
import operator
import timeit
import matplotlib.pyplot as plt 
import scipy.stats as sp
# Enter You Name Here
myname = "Saurav-vara-prasad-Channuri-" # or "Doe-Jane-"

# Implement your decision tree below


def parent_entropy(dataset):    

    total_data_size = len(dataset.T[0])

    # print
    # print
    # print "total_data_size = ", total_data_size
    attribute_size = len(dataset[0])
    # print "attribute_size = ", attribute_size

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
    # print "parent entropy = ", entrpy

    return entrpy

def entropy(dataset, output_vector, threshold):

    zeros_and_ones = 1*dataset
    total_data_size = len(dataset.T[0])
    attribute_size = len(dataset[0])
 
    entorpy_one = 0
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

    # if(child_count ==0 or count_first_child_ones == 0):
    #     ones_entropy = 0

    # if(child_count == 0 or count_first_child_zeros == 0):
    #     zeros_entropy = 0

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

    # print "time taken = ", stop - start

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

    if test_instance[root.attribute_index] > root.threshold:
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
            # print "node threshold = ",root," node attribute = ", node.attribute_index



class DecisionTree():
    tree = {}

    def learn(self, training_set, completed_attributes, root, output_vector, completed_attributes_final, dims, attri_choice):
      
        # print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"


        attr = np.sort(np.random.choice(len(training_set[0])-1, attri_choice, replace =False))
        # print "attr = ", len(attr)
        # print
        # completed_attributes_final = np.random.choice(dims, 8 , replace =False)

        attr = np.append(attr, len(training_set[0])-1)
        # print "attrs = ", attr 
        temp_train = training_set
        node_values = information_gain(temp_train[:, attr], output_vector)
      	
      	# print "len (node values) = ", len(node_values)

        threshold_array = node_values[3]
        threshold_array = np.append(threshold_array, 0)

        threshold = node_values[1]
        attribute = node_values[0]
      
        # print "attribute = ", attribute
        real_attribute = attr[attribute]
        # print "real attr = ", real_attribute

        temp_attributes =np.linspace(0,len(training_set[0])-1, len(training_set))
        # print "temp attributes = ", temp_attributes


        # completed_attributes = np.delete(np.array(temp_attributes).astype(int), attribute, 0)
        # completed_attributes_final = np.delete(np.array(temp_attributes).astype(int), attribute, 0)

        best_information_gain = node_values[2]  
      
        # print "dashdash = ", threshold_array.shape
        left_child = (1*(training_set.T[real_attribute] <= threshold_array[attribute]))
        right_child = (1*(training_set.T[real_attribute] > threshold_array[attribute]))

        left_child_indexes = np.argwhere(left_child > 0).T[0]
        right_child_indexes = np.argwhere(right_child > 0).T[0]
        # print " lenlenelen = ", len(left_child_indexes)

        training_set_temp = training_set[:,completed_attributes]


        left_child_dataset = training_set[left_child_indexes,:]
        right_child_dataset = training_set[right_child_indexes,:]
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

     
        if (np.float(best_information_gain) != 0 ): #and np.count_nonzero(completed_attributes) != 0 and attribute_size > 2):
            # print
            output_vector = output_vector_temp[left_child_indexes]
            root_left = self.learn(left_child_dataset, completed_attributes, None, output_vector,completed_attributes_final, dims, attri_choice)
            root.left = root_left
        if(np.float(best_information_gain) != 0 ):#and np.count_nonzero(completed_attributes) != 0 and attribute_size > 2):
            # print
            output_vector = output_vector_temp[right_child_indexes]
            root_right = self.learn(right_child_dataset, completed_attributes, None, output_vector, completed_attributes_final, dims, attri_choice)
            root.right = root_right

        return root

    # implement this function


    def classify(self, test_instance, root, training_set):

        result = classification(test_instance, root, training_set)
        return result



def run_decision_tree():

    # Load data set
    with open("spambase_data.csv") as f:
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
    # data = np.random.shuffle(data)

    trees = 10

    attri_choice = 8
    attri_scores = []
    training_set = [X1, X2, X3, X4, X5, X6, X7, X8_validation]

    print "Number of records: %d" % len(data)

    # K = 15
    # You need to modify the following code for cross validation.
    # K = len(data) - int(len(data)*0.2)

    # test_set = (data[K:])
    # training_set = (data[:K])

    K = 10
    training_set = [x for i, x in enumerate(data) if (i % K != 9  and i%K !=3 and i%K != 6)]
    test_set = [x for i, x in enumerate(data) if (i % K == 9 or i%K ==3 or i%K == 6) ]
  	
 #  	print "training length = ", len(trainin_set)
	# print "test length = ", len(test_set)
    

    
    print "training len", len(training_set)
    print "test len ", len(test_set)

    axis = []
    axis_accuracy = []

    



    for itera in range(3, len(training_set[0]),5):
	    tree_list = []
	    axis = np.append(axis, itera)

	    datasets = []

	    for j in range(0, trees):
	        print "tree ", j
	        tree = DecisionTree()
	        dims =  len(training_set[0])
	        completed_attributes = np.linspace(0,dims-1,dims).astype(int)
	        # print completed_attributes
	        training_set = np.array(training_set).astype(float)
	        random_data_indices = np.sort(np.random.choice(len(training_set)-1, 500, replace =False))
	        training_set_temp = training_set[ random_data_indices ]
	        print "len = ", len(training_set_temp)

	        root = tree.learn( training_set_temp ,completed_attributes, None,training_set_temp.T[dims-1], completed_attributes, dims, itera)
	        tree_list.append(tree) 
	        datasets.append(training_set_temp)

	    # Classify the test set using the tree we just constructed
	    results = []
	    print "prediction"
	    print "length tree list = ", len(tree_list)
	    accuracy_list = []

	    for instance in training_set:
	        outputs = []
	        for j in range(0, len(tree_list)):            
	             tree = tree_list[j]
	             # print "instance = ",instance 
	             if(  np.sum(1*np.isin(instance, datasets[j])) != 0 ) :
	             	outputs.append(tree.classify( np.array(instance[:-1]).astype(float) ,root, training_set ))
	             	result = sp.mode(outputs)[0][0]
	             	results.append( float(result) == float(instance[dims-1]) )

	    accuracy = float(results.count(True))/float(len(results))
	    axis_accuracy.append(accuracy)
	    print "accuracy: %.4f" % accuracy
	    
    # Writing results to a file (DO NOT CHANGE)
    print "average of accuracies = ", np.mean(axis_accuracy)
    plt.xlabel('shuffle values')
    plt.ylabel('accuracies')
    plt.plot(axis, axis_accuracy)
    plt.show()
    f = open(myname+"result2.txt", "w")
    f.write("accuracy: %.4f" % accuracy)
    f.close()
    return accuracy

if __name__ == "__main__":

    run_decision_tree()
