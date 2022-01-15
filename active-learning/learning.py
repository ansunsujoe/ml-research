import numpy as np
import logging
import torch

def seq_to_array(fp, max_len=15):
    with open(fp) as f:
        data = [x for x in f.read().split("\n") if x != ""]
        new_list = []
        for seq in data:
            arr = seq.split(",")
            new_list.append(np.array([int(w) for w in arr] + [0] * (max_len - len(arr))))
    return np.vstack(new_list)

def seq_match(r, s):
    l_r = len(r)
    l_s = len(s)
    for i in range(l_r - l_s + 1):
        if np.dot(r[i:i+l_s] - s, s) == 0:
            return True
    return False

class Agent:
    def __init__(self):
        pass
    
    def human_label(self, X):
        labels = np.zeros(X.shape[0])
        for r in X:
            if seq_match(r, np.array([1, 0, 5])):
                pass
    
if __name__ == "__main__":
    # print(seq_to_array("sequences-1-test.txt"))
    r = np.array([5, 7, 3, 4, 6, 3, 2, 5, 2, 3, 1, 2, 3, 4])
    s = np.array([0, 2, 3])
    
    