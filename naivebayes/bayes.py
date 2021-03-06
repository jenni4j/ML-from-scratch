from numpy import *

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],['stop', 'posting', 'stupid', 'worthless', 'garbage'],['mr', 'licks', 'ate', 'my', 'steak', 'how','to', 'stop', 'him'],['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my vocabulary!" % word)
    return returnVec

def trainNBO(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    #note that class=1 is abusive document
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    pONum = ones(numWords)
    p1Num = ones(numWords)
    pODenom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            pONum += trainMatrix[i]
            pODenom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    pOVect = log(pONum/pODenom)
    return pOVect,p1Vect,pAbusive

def classifyNB(vec2Classify, pOVec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    pO = sum(vec2Classify * pOVec) + log(1.0 - pClass1)
    if p1 > pO:
        return 1
    else:
        return 0

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) >2]

def spamTest():


