
"""#############################################################
AI Project 2 Feature Selection with Nearest Neighbor

Revision History:
v0.1  read in dataset using numpy and basic command line interface.
v0.2  add Forward Selection and dummy validation
v0.3  add Backward Elimination with dummy validation 
v1.0  add nearest neighbor classifier
      make changes to print out best set and summary
      add print to result file
v1.01 add execution time
v1.02 update the index access in classifier function, same result
v1.1  use numpy function to slice the features into 1-D array, 
      and use numpy function to calculate euclidean distance between 2-arrays
      verfied no impact on results
        
References:
1. project handout on the forward selection 
2. internet on some python usage

DataSets:
My data set:
CS170_Fall_2021_LARGE_data__9.txt
CS170_Fall_2021_SMALL_data__39.txt

Known test data set from project handout: 
-- not sure if it is correct set 
CS170_Fall_2021_LARGE_data__27.txt  (feature 36, 35, 12 accuracy = 97% )
CS170_Fall_2021_SMALL_data__86.txt  (feature  4,  5,  7 accuracy = 93% )
"""

import numpy as np # for array operation, read in data
import random # for generate random (0, 1) for dummy classifier
import os
import math
import copy
import time   # algorithm execution time

# forward selection 
# -- directly mapping from project handout matlab code to python
# start from empty set, and add most relevant feature one at a time
def forward_selection(data, full_feature_set, full_feature_accuracy):
    
    best_feature_set = copy.deepcopy(full_feature_set)
    best_accuracy = full_feature_accuracy
    
    current_set_of_features = []
    row, col = data.shape 
    for i in range(1, col):
        feature_to_add_at_this_level = 0
        best_so_far_accuracy    = 0    
        for k in range(1, col):
            if not k in current_set_of_features: #Only consider adding, if not already added.
                accuracy = leave_one_out_cross_validation(data,current_set_of_features,k, 1)
                if accuracy > best_so_far_accuracy : 
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k                   
        current_set_of_features.append(feature_to_add_at_this_level)
        #update overall best set and accuracy
        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_set = copy.deepcopy(current_set_of_features)
        elif best_so_far_accuracy < best_accuracy:
            print_log("(Warning, Accuracy has decreased! Continue search in case of local maxima)")
        print_log(f'Feature set {current_set_of_features} was best, accuracy is {convert_percentage(best_so_far_accuracy)}\n')
        
    # completing search to return best set and accuracy
    return  (best_set, best_accuracy) 
    
# backward elimination 
# reverse of forward selection
# start from full set, and remove least relevant feature one at a time
def backward_elimination(data, full_feature_set, full_feature_accuracy):
    
    best_feature_set = copy.deepcopy(full_feature_set)
    best_accuracy = full_feature_accuracy
    
    row, col = data.shape
    current_set_of_features = list(range(1, col))  # generate full feature list 
    for i in range(1, col-1):
        feature_to_remove_at_this_level = 0
        best_so_far_accuracy = 0   
        for k in range(1, col):
            if k in current_set_of_features: #Only consider removing, if not already removed.
                accuracy = leave_one_out_cross_validation(data,current_set_of_features,k, 0)  
                if accuracy > best_so_far_accuracy : 
                    best_so_far_accuracy = accuracy
                    feature_to_remove_at_this_level = k                   
        current_set_of_features.remove(feature_to_remove_at_this_level) # has feature list only with unique 
        #update overall best set and accuracy
        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_set = copy.deepcopy(current_set_of_features)
        elif best_so_far_accuracy < best_accuracy:
            print_log("(Warning, Accuracy has decreased! Continue search in case of local maxima)\n")

        print_log(f'Feature set {current_set_of_features} was best, accuracy is {convert_percentage(best_so_far_accuracy)}\n')        

    # completing search to return best set and accuracy
    return  (best_set, best_accuracy)   
        
# dummy classifier return random float number [0,1)       
def leave_one_out_cross_validation(data, current_set, feature_to_change, changeflag):
    #return random.random() 
    row, col = data.shape 
    feature_to_validate = copy.deepcopy(current_set)
    if changeflag == 1 and feature_to_change != 0: #add feature for cross validation
        feature_to_validate.append(feature_to_change)
    elif changeflag == 0 and feature_to_change != 0: # eliminate feature for cross validation
        feature_to_validate.remove(feature_to_change)

    if len(feature_to_validate) == 0: #if an empty feature set
        return 0  
        
    number_correctly_classfied = 0;
    
    for i in range(0, row): #loop through all instances (each row)
        #1-D array [#features] of instance i (row i)
        object_to_classify = np.take(data[i], feature_to_validate, 0) 
        label_object_to_classify = data[i][0]
        nearest_neighbor_distance = float('inf') # set to max number
        nearest_neighbor_location = row +1 # set to max instance

        for k in range(0, row): 
            #loop through all other instances (each row) to find closest neighbor
            if k != i:
                #1-D array [features] of instance k (row k)
                target = np.take(data[k], feature_to_validate, 0)
                #euclidean distance between two arrays
                distance = np.linalg.norm(object_to_classify - target)
                if  distance <   nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label    = data[nearest_neighbor_location][0] 
        # increment if correctly classified            
        if label_object_to_classify == nearest_neighbor_label: 
            number_correctly_classfied += 1
    accuracy = number_correctly_classfied / row
    print_log(f'       Using feature(s) {feature_to_validate} accuracy is {convert_percentage(accuracy)}') 
    return accuracy     

def process_input():
    print_log ("Project 2: Feature Selection Algorithm")
    testfile = input("Enter the test file name:")  
    print_log(f'Data file entered {testfile}')
    if not os.path.isfile(testfile):
        print_log("data file not exist\n")
        exit(1)
        
    alg = input("Enter the algorithm you want to run.\n" +
                        "  Type '1' for Forward Selection\n  Type '2' for Backward Elimination \n" )   
    print_log(f'algorithm option entered {alg}')                          
    if alg != "1"  and alg != "2":
        print_log("wrong algorithm entered\n")
        exit(1) 
        
    return (testfile, alg)
        
def convert_percentage(accuracy):
    return '{:2.2%}'.format(accuracy)

def print_log(resultstr):
    #print to screen
    print(resultstr)
    
    #print to file
    resfile = open('result.txt', 'a')
    resfile.writelines(resultstr)
    resfile.writelines("\n")   
    resfile.close()    

# main driver
if __name__ == "__main__":
    # process input to get testfile name and algorithm to use
    testfile, alg = process_input()    
    
    #load text file into array with numpy
    #assume data file with right format
    arr = np.loadtxt(testfile)
    row, col = arr.shape    
    numFeatures = col -1
    numInstances = row
    
    print_log(f'The dataset has {numFeatures} features (not including the class attribute), with {numInstances} instances.\n')

    #calculate accuracy with all features
    full_set = list(range(1, col))
    accuracy = leave_one_out_cross_validation(arr, full_set, 0, 2)    
    print_log(f'Running nearest neighbor with all {numFeatures}, using "leaving-one-out" evaluation, I get an accuracy of {convert_percentage(accuracy)}\n')    

    start_time = time.time()    
    if alg == "1": # use forward selection algorithm
        print_log("Beginning search with Forward Selection,\n")
        best_set, best_accuracy = forward_selection(arr, full_set, accuracy)
    elif alg == "2": # use reverse elimination algorithm
        print_log ("Beginning search with Backward Elimination,\n")
        best_set, best_accuracy = backward_elimination(arr, full_set, accuracy)
    
    end_time = time.time() 
    elapsed_time = end_time - start_time;  # execution time
    print_log("search execution time: " + str(round(elapsed_time, 4)) + "s" + "\n")  
    
    print_log(f'Finished Search!! The best feature subset is {best_set}, which has an accuracy of {convert_percentage(best_accuracy)}') 

    
"""
sample output v04:
python project2_ai.py
Project 2: Feature Selection Algorithm
Enter the test file name:small_61.txt
Enter the algorithm you want to run.
  Type '1' for Forward Selection
  Type '2' for Backward Elimination
1
The dataset has 10 features (not including the class attribute), with 500 instances.

       Using feature(s) [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] accuracy is 78.00%
Running nearest neighbor with all 10, using "leaving-one-out" evaluation, I get an accuracy of 78.00%

Beginning search with Forward Selection,

       Using feature(s) [1] accuracy is 72.80%
       Using feature(s) [2] accuracy is 74.40%
       Using feature(s) [3] accuracy is 74.20%
       Using feature(s) [4] accuracy is 71.60%
       Using feature(s) [5] accuracy is 83.80%
       Using feature(s) [6] accuracy is 71.60%
       Using feature(s) [7] accuracy is 72.80%
       Using feature(s) [8] accuracy is 76.20%
       Using feature(s) [9] accuracy is 72.80%
       Using feature(s) [10] accuracy is 75.80%
Feature set [5] was best, accuracy is 83.80%

       Using feature(s) [5, 1] accuracy is 83.00%
       Using feature(s) [5, 2] accuracy is 80.60%
       Using feature(s) [5, 3] accuracy is 84.80%
       Using feature(s) [5, 4] accuracy is 83.00%
       Using feature(s) [5, 6] accuracy is 85.40%
       Using feature(s) [5, 7] accuracy is 83.40%
       Using feature(s) [5, 8] accuracy is 86.40%
       Using feature(s) [5, 9] accuracy is 84.00%
       Using feature(s) [5, 10] accuracy is 98.20%
Feature set [5, 10] was best, accuracy is 98.20%

       Using feature(s) [5, 10, 1] accuracy is 92.80%
       Using feature(s) [5, 10, 2] accuracy is 91.60%
       Using feature(s) [5, 10, 3] accuracy is 93.20%
       Using feature(s) [5, 10, 4] accuracy is 91.20%
       Using feature(s) [5, 10, 6] accuracy is 93.80%
       Using feature(s) [5, 10, 7] accuracy is 91.60%
       Using feature(s) [5, 10, 8] accuracy is 91.40%
       Using feature(s) [5, 10, 9] accuracy is 92.80%
(Warning, Accuracy has decreased! Continue search in case of local maxima)
Feature set [5, 10, 6] was best, accuracy is 93.80%

       Using feature(s) [5, 10, 6, 1] accuracy is 89.20%
       Using feature(s) [5, 10, 6, 2] accuracy is 88.00%
       Using feature(s) [5, 10, 6, 3] accuracy is 91.20%
       Using feature(s) [5, 10, 6, 4] accuracy is 88.80%
       Using feature(s) [5, 10, 6, 7] accuracy is 88.00%
       Using feature(s) [5, 10, 6, 8] accuracy is 87.20%
       Using feature(s) [5, 10, 6, 9] accuracy is 90.40%
(Warning, Accuracy has decreased! Continue search in case of local maxima)
Feature set [5, 10, 6, 3] was best, accuracy is 91.20%

       Using feature(s) [5, 10, 6, 3, 1] accuracy is 88.00%
       Using feature(s) [5, 10, 6, 3, 2] accuracy is 90.00%
       Using feature(s) [5, 10, 6, 3, 4] accuracy is 85.40%
       Using feature(s) [5, 10, 6, 3, 7] accuracy is 87.80%
       Using feature(s) [5, 10, 6, 3, 8] accuracy is 87.60%
       Using feature(s) [5, 10, 6, 3, 9] accuracy is 87.80%
(Warning, Accuracy has decreased! Continue search in case of local maxima)
Feature set [5, 10, 6, 3, 2] was best, accuracy is 90.00%

       Using feature(s) [5, 10, 6, 3, 2, 1] accuracy is 84.60%
       Using feature(s) [5, 10, 6, 3, 2, 4] accuracy is 83.00%
       Using feature(s) [5, 10, 6, 3, 2, 7] accuracy is 85.60%
       Using feature(s) [5, 10, 6, 3, 2, 8] accuracy is 83.60%
       Using feature(s) [5, 10, 6, 3, 2, 9] accuracy is 85.60%
(Warning, Accuracy has decreased! Continue search in case of local maxima)
Feature set [5, 10, 6, 3, 2, 7] was best, accuracy is 85.60%

       Using feature(s) [5, 10, 6, 3, 2, 7, 1] accuracy is 82.20%
       Using feature(s) [5, 10, 6, 3, 2, 7, 4] accuracy is 80.60%
       Using feature(s) [5, 10, 6, 3, 2, 7, 8] accuracy is 81.00%
       Using feature(s) [5, 10, 6, 3, 2, 7, 9] accuracy is 81.60%
(Warning, Accuracy has decreased! Continue search in case of local maxima)
Feature set [5, 10, 6, 3, 2, 7, 1] was best, accuracy is 82.20%

       Using feature(s) [5, 10, 6, 3, 2, 7, 1, 4] accuracy is 76.20%
       Using feature(s) [5, 10, 6, 3, 2, 7, 1, 8] accuracy is 80.60%
       Using feature(s) [5, 10, 6, 3, 2, 7, 1, 9] accuracy is 80.00%
(Warning, Accuracy has decreased! Continue search in case of local maxima)
Feature set [5, 10, 6, 3, 2, 7, 1, 8] was best, accuracy is 80.60%

       Using feature(s) [5, 10, 6, 3, 2, 7, 1, 8, 4] accuracy is 76.40%
       Using feature(s) [5, 10, 6, 3, 2, 7, 1, 8, 9] accuracy is 79.60%
(Warning, Accuracy has decreased! Continue search in case of local maxima)
Feature set [5, 10, 6, 3, 2, 7, 1, 8, 9] was best, accuracy is 79.60%

       Using feature(s) [5, 10, 6, 3, 2, 7, 1, 8, 9, 4] accuracy is 78.00%
(Warning, Accuracy has decreased! Continue search in case of local maxima)
Feature set [5, 10, 6, 3, 2, 7, 1, 8, 9, 4] was best, accuracy is 78.00%

Finished Search!! The best feature subset is [5, 10], which has an accuracy of 98.20%


Sample output v03:
python project2_ai.py
Project 2: Feature Selection Algorithm
Enter the test file name:small.txt
Enter the algorithm you want to run.
  Type '1' for Forward Selection
  Type '2' for Backward Elimination
2
The dataset has 10 features (not including the class attribute), with 500 instances.

[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
On the 1 th level of the search tree
--Considering removing the 1 feature
--Considering removing the 2 feature
--Considering removing the 3 feature
--Considering removing the 4 feature
--Considering removing the 5 feature
--Considering removing the 6 feature
--Considering removing the 7 feature
--Considering removing the 8 feature
--Considering removing the 9 feature
--Considering removing the 10 feature
On level 1 removed feature 9 from current set
On the 2 th level of the search tree
--Considering removing the 1 feature
--Considering removing the 2 feature
--Considering removing the 3 feature
--Considering removing the 4 feature
--Considering removing the 5 feature
--Considering removing the 6 feature
--Considering removing the 7 feature
--Considering removing the 8 feature
--Considering removing the 10 feature
On level 2 removed feature 6 from current set
On the 3 th level of the search tree
--Considering removing the 1 feature
--Considering removing the 2 feature
--Considering removing the 3 feature
--Considering removing the 4 feature
--Considering removing the 5 feature
--Considering removing the 7 feature
--Considering removing the 8 feature
--Considering removing the 10 feature
On level 3 removed feature 3 from current set
On the 4 th level of the search tree
--Considering removing the 1 feature
--Considering removing the 2 feature
--Considering removing the 4 feature
--Considering removing the 5 feature
--Considering removing the 7 feature
--Considering removing the 8 feature
--Considering removing the 10 feature
On level 4 removed feature 8 from current set
On the 5 th level of the search tree
--Considering removing the 1 feature
--Considering removing the 2 feature
--Considering removing the 4 feature
--Considering removing the 5 feature
--Considering removing the 7 feature
--Considering removing the 10 feature
On level 5 removed feature 7 from current set
On the 6 th level of the search tree
--Considering removing the 1 feature
--Considering removing the 2 feature
--Considering removing the 4 feature
--Considering removing the 5 feature
--Considering removing the 10 feature
On level 6 removed feature 2 from current set
On the 7 th level of the search tree
--Considering removing the 1 feature
--Considering removing the 4 feature
--Considering removing the 5 feature
--Considering removing the 10 feature
On level 7 removed feature 1 from current set
On the 8 th level of the search tree
--Considering removing the 4 feature
--Considering removing the 5 feature
--Considering removing the 10 feature
On level 8 removed feature 5 from current set
On the 9 th level of the search tree
--Considering removing the 4 feature
--Considering removing the 10 feature
On level 9 removed feature 4 from current set
On the 10 th level of the search tree
--Considering removing the 10 feature
On level 10 removed feature 10 from current set

sameple output v0.2:
python project2_ai.py
Project 2: Feature Selection Algorithm
Enter the test file name:CS170_Fall_2021_SMALL_data__86.txt
Enter the algorithm you want to run.
Type '1' for Forward Selection
Type '2' for Backward Elimination
1
Total row 500
Total col 11
The dataset has 10 features (not including the class attribute), with 500 instances.

On the 1 th level of the search tree
--Considering adding the 1 feature
--Considering adding the 2 feature
--Considering adding the 3 feature
--Considering adding the 4 feature
--Considering adding the 5 feature
--Considering adding the 6 feature
--Considering adding the 7 feature
--Considering adding the 8 feature
--Considering adding the 9 feature
--Considering adding the 10 feature
On level 1 added feature 1 to current set
On the 2 th level of the search tree
--Considering adding the 2 feature
--Considering adding the 3 feature
--Considering adding the 4 feature
--Considering adding the 5 feature
--Considering adding the 6 feature
--Considering adding the 7 feature
--Considering adding the 8 feature
--Considering adding the 9 feature
--Considering adding the 10 feature
On level 2 added feature 3 to current set
On the 3 th level of the search tree
--Considering adding the 2 feature
--Considering adding the 4 feature
--Considering adding the 5 feature
--Considering adding the 6 feature
--Considering adding the 7 feature
--Considering adding the 8 feature
--Considering adding the 9 feature
--Considering adding the 10 feature
On level 3 added feature 7 to current set
On the 4 th level of the search tree
--Considering adding the 2 feature
--Considering adding the 4 feature
--Considering adding the 5 feature
--Considering adding the 6 feature
--Considering adding the 8 feature
--Considering adding the 9 feature
--Considering adding the 10 feature
On level 4 added feature 9 to current set
On the 5 th level of the search tree
--Considering adding the 2 feature
--Considering adding the 4 feature
--Considering adding the 5 feature
--Considering adding the 6 feature
--Considering adding the 8 feature
--Considering adding the 10 feature
On level 5 added feature 2 to current set
On the 6 th level of the search tree
--Considering adding the 4 feature
--Considering adding the 5 feature
--Considering adding the 6 feature
--Considering adding the 8 feature
--Considering adding the 10 feature
On level 6 added feature 4 to current set
On the 7 th level of the search tree
--Considering adding the 5 feature
--Considering adding the 6 feature
--Considering adding the 8 feature
--Considering adding the 10 feature
On level 7 added feature 6 to current set
On the 8 th level of the search tree
--Considering adding the 5 feature
--Considering adding the 8 feature
--Considering adding the 10 feature
On level 8 added feature 10 to current set
On the 9 th level of the search tree
--Considering adding the 5 feature
--Considering adding the 8 feature
On level 9 added feature 8 to current set
On the 10 th level of the search tree
--Considering adding the 5 feature
On level 10 added feature 5 to current set


sample output v0.1:
python project2_ai.py
Project 2: Feature Selection Algorithm
Enter the test file name:CS170_Fall_2021_LARGE_data__9.txt
Enter the algorithm you want to run.
Type '1' for Forward Selection
Type '2' for Backward Elimination
1
Total row 2000
Total col 51
The dataset has 50 features (not including the class attribute), with 2000 instances.

"""
   
