import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import operator

from import_statics import file2matrix

dating_date, dating_labels = file2matrix("knn_example_dating/dating.txt")

'''
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dating_date[:,0], dating_date[:,1], 15.0* np.array(dating_labels), 15.0*np.array(dating_labels))
plt.show()
'''

def norm_datefun(dateset):
    min_vals = dateset.min(0)
    max_vals = dateset.max(0)
    range = max_vals - min_vals

    norm_date = np.zeros(np.shape(dateset))
    m = norm_date.shape[0]
    norm_date = dateset - np.tile(min_vals, (m,1))
    norm_date = norm_date / np.tile(range, (m,1))
    return norm_date, min_vals, range


def classify_date(inx, dateset, labels, k):
    norm_date, min_vals, range_diff = norm_datefun(dateset)
    norn_inx = (inx-min_vals) / range_diff

    m = norm_date.shape[0]
    diff = np.tile(norn_inx,(m,1)) - norm_date
    sq_diff = diff**2
    sum_sq_diff = sq_diff.sum(axis = 1)
    distances = sum_sq_diff**0.5

    sorted_indices = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_indices[i]]
        class_count[vote_label] = class_count.get(vote_label,0) + 1
    sorted_class_count = sorted(class_count.items(), key = operator.itemgetter(1),reverse = True)
    return sorted_class_count[0][0]

'''
test = np.array([2000,50,20])
label = classify_date(test, dating_date,dating_labels,2)
print(label)
'''