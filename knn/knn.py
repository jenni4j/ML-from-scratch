from numpy import *
import operator
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

# k-nearest neighbors algorithm
# inX: input vector to classify
# dataSet: full matrix of training examples
# labels: vector of labels (should have as many elements in it as there are rows in dataSet matrix)
# k: number of nearest neighbors to use in voting

def classifyO(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # distance calculation
    diffMat = tile(inX, (dataSetSize,1)) - dataSet 
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort() # returns position of element it WOULD sort
    # voting with lowest k distances
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    # sort dictionary
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
