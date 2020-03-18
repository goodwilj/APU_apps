import math
import mnist
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics


def transpose(a):
    result = [[0 for r in range(len(a))] for c in range(len(a[0]))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            result[j][i]=a[i][j]
    return result

def normalize(a):
    maxval = 255
    for i in range(len(a)):
        for j in range(len(a[0])):
            a[i][j] = a[i][j]/maxval
    return a

def flatten(a):
    result = [0 for c in range(len(a)*len(a[0]))]
    index = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            result[index] = a[i][j]
            index+=1
    return result

digits = mnist.train_images();
n_samples = len(digits)
data = digits.reshape((n_samples, -1))
targets = mnist.train_labels();
binarydata = []
binarytargets = []

batchMax = 1000
batch = 0
for i in range(len(targets)):
    if ((targets[i]==0 or targets[i]==1) and batch < batchMax):
        binarydata.append(data[i])
        binarytargets.append(targets[i])
        batch += 1
        
classifier = svm.SVC(gamma=.001)

classifier.fit(binarydata,binarytargets)
supvec = classifier.support_vectors_
dualcoeffs = classifier.dual_coef_
dualcoeffs = dualcoeffs[0]
intercept = classifier.intercept_

tests = mnist.test_images();
n_samples = len(tests)
data2 = tests.reshape((n_samples, -1))
testtargets = mnist.test_labels();
binarytest = []
binarytesttargets = []

batch = 0
for i in range(len(testtargets)):
    if ((testtargets[i]==0 or testtargets[i]==1) and batch < batchMax):
        binarytest.append(data2[i])
        binarytesttargets.append(testtargets[i])
        batch += 1


Z = flatten(transpose(supvec.tolist()))
X = flatten(binarytest)

with open("Xmatrix.bin", 'wb') as output:
    output.write(bytearray(i for i in X))

with open("Zmatrix.bin", 'wb') as output:
    output.write(bytearray(int(i) for i in Z))

with open("DualCoeffs.bin", 'wb') as output:
    output.write(bytearray(int(i+1) for i in dualcoeffs))

with open("TestTargets.bin", 'wb') as output:
    output.write(bytearray(i for i in binarytesttargets))

    
               
            
