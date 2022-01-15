import random
import logging

# Logging handler initialization
FORMATTER = logging.Formatter("%(message)s")
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# Logging handler setup
logger.setLevel(logging.DEBUG)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(FORMATTER)

# An episode of the stock game
prices = [2, 4, 6, 7, 3, 7, 5, 3, 4]
sar_triples = []

# Starting variables
buy_price = None
bought = False
decision = None
total_reward = 0
cur_price = 0
cur_triple = 0

# A single episode run
for i in range(8):
    # Update price
    cur_price = prices[i]
    
    # Add any rewards to (S, A, R) triples as a result of the
    # price change.
    
    
    # Policy to come up with the decision
    if not bought:
        if random.random() < 0.5:
            decision = "BUY"
        else:
            decision = "ABSTAIN"
    else:
        if random.random() < 0.5:
            decision = "SELL"
        else:
            decision = "HOLD"
    
    # Execution of the decision
    if decision == "BUY":
        # Change the environment
        buy_price = cur_price
        bought = True
        
        # Create (S, A, R) triplet
        state = cur_price
        sar_triples.append([state, 0, 0])
        
    elif decision == "SELL":
        # Change the environment
        total_reward += cur_price - buy_price
        bought = False
        buy_price = None
        
        # Create (S, A, R) triplet
        state = cur_price
        sar_triples.append([state, 1, 0])
        
    elif decision == "ABSTAIN":
        pass
        
    elif decision == "HOLD":
        pass

# If the agent owns the stock after the episode is over,
# add the net gains.
if bought:
    total_reward += cur_price - buy_price
    bought = False
    buy_price = None

logger.info(f"Total reward is ${total_reward}")