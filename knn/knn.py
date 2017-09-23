# k-nearest neighbors algorithm
# For every point in our dataset:
# calculate the distance between inX and the current point
# sort the distances in increasing order
# take k items with lowest distances to inX
# find the majority class among these items
# return the majority class as our prediction for the class of inX


from numpy import *
import operator
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# inX: input vector to classify
# dataSet: full matrix of training examples
# labels: vector of labels (should have as many elements in it as there are rows in dataSet matrix)
# k: number of nearest neighbors to use in voting

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # distance calculation
    diffMat = tile(inX, (dataSetSize,1)) - dataSet 
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort() # returns position of element that puts array in sorted order
    # voting with lowest k distances
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    # sort dictionary
    sortedClassCount = sorted(classCount, key=classCount.__getitem__,reverse=True)
    return sortedClassCount[0]

def file2matrix(filename):
    file = open(filename)
    num_lines = len(file.readlines())
    return_matrix = zeros((num_lines,3))
    class_label_vector = []
    file = open(filename)
    index = 0
    for line in file.readlines():
        line = line.strip()
        list_from_line = line.split('\t')
        return_matrix[index,:] = list_from_line[0:3]
        class_label_vector.append(list_from_line[-1])
        index += 1
    return return_matrix, class_label_vector

def autoNorm(dataset):
    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    norm_dataset = zeros(shape(dataset))
    m = dataset.shape[0]
    # use tile() function to create a matrix the same size as input matrix and then fill up with many copies
    # input matrix 1000x3, min_vals 1x3
    norm_dataset = dataset - tile(min_vals,(m,1))
    norm_dataset = norm_dataset/tile(ranges,(m,1))
    return norm_dataset, ranges, min_vals

def datingClassTest():
    ratio = 0.10
    dating_data_Mat,dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = autoNorm(dating_data_mat)
    print(norm_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m*ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(norm_mat[i,:],norm_mat[num_test_vecs:m,:],dating_labels[num_test_vecs:m],3)
        print("the classifier came back with: %s, the real answer is: %s" % (classifier_result, dating_labels[i]))
        if (classifier_result != dating_labels[i]): 
            error_count += 1.0
    print("the total error rate is: %f" % (error_count/float(num_test_vecs)))

def classifyPerson():
    result_list = ['Run away', 'Ok for a drink', 'Get married']
    percent_vg = float(input("% of time spent playing video games?"))
    ff_miles = float(input("frequent flier miles earned/year?"))
    ice_cream = float(input("litres of ice cream consumed/year?"))
    dating_data_Mat,dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = autoNorm(dating_data_mat)
    in_arr = array([ff_miles, percent_vg, ice_cream])
    classifier_result = classify0((in_arr-min_vals)/ranges,norm_mat,dating_labels,3)
    print("You will probably like this person: ",result_list[classifier_result-1])
    

if __name__ == '__main__':
    group, labels = createDataSet()
    results = classify0([0,0],group,labels,3)
    dating_data_mat, dating_labels = file2matrix('datingTestSet.txt')
    norm_mat, ranges, min_vals = autoNorm(dating_data_mat)
    datingClassTest()
    classifyPerson()


