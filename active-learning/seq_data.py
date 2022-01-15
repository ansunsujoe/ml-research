import random
from tqdm import tqdm

def random_seq():
    return [str(random.randint(1, 9)) for x in range(random.randint(2, 15))]

if __name__ == "__main__":
    with open("sequences-1-train.txt", "w") as f:
        for i in tqdm(range(5000)):
            f.write(",".join(random_seq()) + "\n")