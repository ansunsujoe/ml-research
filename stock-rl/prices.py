import random
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from collections import deque

# Logging handler initialization
FORMATTER = logging.Formatter("%(message)s")
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Logging handler setup
logger.setLevel(logging.DEBUG)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(FORMATTER)

# Simple neural network
class FFNN(nn.Module):
    """
    Feed forward neural network with one hidden layer
    """
    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 4)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Memory:
    """
    Similar to the RingBuffer data structure in Keras RL.
    https://github.com/keras-rl/keras-rl/blob/216c3145f3dc4d17877be26ca2185ce7db462bad/rl/memory.py#L45
    """
    def __init__(self, max_size):
        self.max_size = max_size
        self.content = deque(max_len=max_size)
    
    def __getitem__(self, idx):
        return self.content[idx]
    
    def append(self, o):
        self.content.append(o)
        
    def __len__(self):
        return len(self.content)
    
class ExpMemory:
    """
    Similar to Keras's SequentialMemory in Keras RL.
    """
    def __init__(self, max_size):
        self.s = Memory(max_size)
        self.a = Memory(max_size)
        self.r = Memory(max_size)
        self.s_next = Memory(max_size)
        
    def __getitem__(self, idx):
        return (self.s[idx], self.a[idx], self.r[idx], 
                self.s_next[idx])
        
    def set_prev_s(self, val):
        self.s[-1] = val
    
    def set_prev_a(self, val):
        self.a[-1] = val
    
    def set_prev_r(self, val):
        self.r[-1] = val
        
    def set_prev_s_prime(self, val):
        self.s_next[-1] = val

if __name__ == "__main__":
    # An episode of the stock game
    prices = [2, 4, 6, 7, 3, 7, 5, 3, 4]
    experiences = ExpMemory(max_size=200)

    # Starting variables
    buy_price = None
    bought = False
    action = None
    prev_action = None
    cur_state = None
    prev_state = None
    episode_reward = 0
    cur_price = 0
    cur_triple = 0
    X = None
    y = None
    
    # Initialize the PyTorch neural network
    q_net = FFNN()
    target_net = FFNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=0.01)

    # A single episode run
    for i in range(8):
        # Update price
        cur_price = prices[i]
        cur_state = [prices[i]]
        
        # Add any rewards to (S, A, R) triples as a result of the
        # price change.
        if i > 0:
            price_change = prices[i] - prices[i-1]
            if prev_action == 0 or prev_action == 3:
                experiences[i-1][-1] = price_change
            else:
                experiences[i-1][-1] = -price_change
            cur_triple += 1
        
        # Policy to come up with the action
        if not bought:
            if random.random() < 0.5:
                action = "BUY"
            else:
                action = "ABSTAIN"
        else:
            if random.random() < 0.5:
                action = "SELL"
            else:
                action = "HOLD"
        
        # Execution of the action
        if action == "BUY":
            # Change the environment
            buy_price = cur_price
            bought = True
            
            # Create (S, A, R) triplet
            state = cur_price
            experiences.append([state, 1, 0, 0, 0, 0])
            
        elif action == "SELL":
            # Change the environment
            episode_reward += cur_price - buy_price
            bought = False
            buy_price = None
            
            # Create (S, A, R) triplet
            state = cur_price
            experiences.append([state, 0, 1, 0, 0, 0])
            
        elif action == "ABSTAIN":
            # Create (S, A, R) triplet
            state = cur_price
            experiences.append([state, 0, 0, 1, 0, 0])
            
        elif action == "HOLD":
            # Create (S, A, R) triplet
            state = cur_price
            experiences.append([state, 0, 0, 0, 1, 0])
            
        # Set action as previous action
        prev_action = action

    # If the agent owns the stock after the episode is over,
    # add the net gains.
    if bought:
        episode_reward += cur_price - buy_price
        bought = False
        buy_price = None

    logger.info(f"Total reward for this episode is ${episode_reward}")
    # logger.info(experiences)
    
    # Get the triples from the episode and preprocess into
    # training data.
    sar_data = np.array(experiences[:-1])
    episode_X = sar_data[:,:-1]
    episode_y = sar_data[:,-1]
    
    # Add episode training data to larger array
    if X:
        X = np.vstack(X, episode_X)
        y = np.vstack(y, episode_y)
    else:
        X = episode_X.copy()
        y = episode_y.copy().reshape(-1, 1)
        
    # Convert to torch DataLoader
    X_tensor = torch.Tensor(X)
    y_tensor = torch.Tensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_dl = DataLoader(dataset)
    
    # Train model on dataset
    for epoch in range(10):
        total_loss = 0.0
        for x, y in train_dl:
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward propagation and backward propagation
            out = q_net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            # Increment loss
            total_loss += loss
            
        # Stats and loss on training loop
        logger.debug(f"Epoch {epoch + 1} - Total Loss: {total_loss}")
        total_loss = 0.0