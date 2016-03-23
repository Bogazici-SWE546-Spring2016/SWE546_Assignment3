import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import heapq as hp

class IrisModel:
    def __init__(self, file_name, seperator=' '):
        data = pd.read_csv(u'data/{0}'.format(file_name), sep=seperator)

        feature1 = np.matrix(data.sl).T
        feature2 = np.matrix(data.sw).T
        feature3 = np.matrix(data.pl).T
        feature4 = np.matrix(data.pw).T

        self.X = np.hstack((feature1, feature2, feature3, feature4))
        self.c = np.matrix(data.c).T

class tup:
    def __init__(self, val, idx):
        self.val = val
        self.idx = idx

    def __lt__(self, other):
        '''Redefine for max-heap'''
        return self.val > other.val

    def __le__(self, other):
        return self.val <= other.val

    def __eq__(self, other):
        return self.val == other.val

    def __ne__(self, other):
        return self.val != other.val

    def __gt__(self, other):
        return self.val > other.val

    def __ge__(self, other):
        return self.val >= other.val

    def __str__(self):
        return '{:.3},{:d}'.format(self.val, self.idx)

def findMaxOccurrence(heap, c):
    categories = []
    for t in range(len(heap)):
        h = hp.heappop(heap)
        categories.append(int(c[h.idx]))
    return max(set(categories), key=categories.count)

def categoryPredictionK_NN(K, A, test, c):
    heap = []
    N = A.shape[0]

    for i in range(K):
        hp.heappush(heap, tup(np.inf, -1))

    for i in range(N):
        e = A[i, :] - test
        e = e.reshape((4, 1))
        tp = tup(float(e.T * e), i)
        if tp <= heap[0]:
            hp.heapreplace(heap, tp)

    return findMaxOccurrence(heap, c)

def categoryPrediction(K, A, c, test_category):
    true_pos  = 0
    true_neg  = 0
    false_pos = 0
    false_neg = 0

    for i in range(50 * (test_category - 1), 100 * (test_category - 1)):
        predicted_category = categoryPredictionK_NN(K, np.delete(A, i, axis=0), A[i, :], c)

        if predicted_category == int(c[i]):
            if int(c[i]) == test_category:
                true_pos += 1
            else:
                true_neg += 1
        else:
            if int(c[i]) == test_category:
                false_pos += 1
            else:
                false_pos += 1

    accuracy = (100. * (true_pos + true_neg)) / (true_pos + true_neg + false_pos + false_neg)
    precision = (100. * true_pos) / (true_pos + false_pos)
    recall = (100. * true_pos) / (true_pos + false_neg)

    return accuracy, precision, recall


K = 3
test = np.mat([1.8, 2.1, 1.3, 1.2])
model = IrisModel('iris.txt')

result = categoryPredictionK_NN(K, model.X, test, model.c)
accuracy, precision, recall  = categoryPrediction(K, model.X, model.c, 2)

print("Category: %d" % result)
print("Accuracy: %f, Precision: %f, Recall: %f" % (accuracy, precision, recall))
