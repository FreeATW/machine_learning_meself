# a example of KNN from https://github.com/apachecn/ailearning/blob/master/docs/ml/2.md
import numpy as np

def file2matrix(filename):
    fr = open(filename,'r')
    num_lines = len(fr.readlines()) 
    return_Mat = np.zeros((num_lines, 3))
    return_labels = []

    fr = open(filename,'r')
    index = 0
    for line in fr.readlines():
        line = line.strip()
        list_from_line = line.split('\t')
        return_Mat[index:] = list_from_line[0:3]
        return_labels.append(int(list_from_line[-1]))
        index +=1
    
    fr.close()

    return return_Mat, return_labels
