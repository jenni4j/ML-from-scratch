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
    index = 0
    for line in file.readlines():
        line = line.strip()
        list_from_line = line.split('\t')
        return_matrix[index,:] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_matrix, class_label_vector
    

if __name__ == "__main__":
    group, labels = createDataSet()
    results = classify0([0,0],group,labels,3)


