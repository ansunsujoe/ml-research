from pathlib import Path
from nltk.tokenize import wordpunct_tokenize
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Vectorizer:
    def __init__(self):
        self.dictionary = {}
        self.next_token = 1
    
    def fit(self, corpus):
        for sent in corpus:
            for word in sent:
                if self.dictionary.get(word) is None:
                    self.dictionary["word"] = self.next_token
                    self.next_token += 1
    
    def fit_transform(self, corpus):
        pass

def process_code_file(fp):
    """
    Process a text file with code in it. Processing is done using
    NLTK.
    """
    with open(fp) as f:
        processed_lines = []
        code_lines = f.readlines()
        for line in code_lines:
            processed_lines.append(wordpunct_tokenize(line.strip()))
        return processed_lines

if __name__ == "__main__":
    # Process file text
    fp = Path("code-data/code-1.txt")
    print(process_code_file(fp))