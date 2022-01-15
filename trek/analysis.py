import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

sentences = []
labels = []

with open("train.txt") as f:
    for r in f.readlines():
        x = r.split()
        labels.append(x[0].split(":")[0])
        sentences.append(" ".join(x[1:]))

# Label encoding for the labels
le = LabelEncoder()
le.fit(labels)
print(le.transform(labels))
print(le.classes_)
print(sentences)