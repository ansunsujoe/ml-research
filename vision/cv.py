from re import A
import numpy as np

def linear_scaling(hist, A, B):
    print(min(hist.keys()))

def qerror(hist, threshold):
    q1 = 0.0
    q1_count = 0
    q2 = 0.0
    q2_count = 0
    
    # Pre-step to calculate means
    for k, v in hist.items():
        if k < threshold:
            q1 += v * k
            q1_count += v
        else:
            q2 += v * k
            q2_count += v
            
    # Calculate means
    q1 = q1 / q1_count if q1_count > 0 else 0
    q2 = q2 / q2_count if q2_count > 0 else 0
    
    # Initialize and calculate error value
    q_error = 0.0
    for k, v in hist.items():
        if k < threshold:
            q_error += (k - q1) ** 2 * v
        else:
            q_error += (k - q2) ** 2 * v
    return q_error

def optimal_threshold(hist):
    q1 = 0.0
    q2 = 255.0
    threshold = -1
    iterations = 0
    
    # Iterations
    while True:
        # Set variables at the beginning of the loop
        q1_sum = 0.0
        q1_count = 0
        q1_max = q1
        q2_sum = 0.0
        q2_count = 0
        q2_min = q2
        midpoint = (q1 + q2) / 2.0
    
        # Pre-step to calculate means
        for k, v in hist.items():
            if k < midpoint:
                q1_sum += v * k
                q1_count += v
                if k > q1_max:
                    q1_max = k
            else:
                q2_sum += v * k
                q2_count += v
                if k < q2_min:
                    q2_min = k
                
        # Calculate means
        q1 = q1_sum / q1_count if q1_count > 0 else 0
        q2 = q2_sum / q2_count if q2_count > 0 else 0
        
        # Increment iterations
        iterations += 1
        
        # Check if threshold is the same as in the last iteration
        if threshold and (q2_min + q1_max) / 2.0 == threshold:
            print(f"Found threshold in {iterations} iterations.")
            print(f"Q1 = {q1}")
            print(f"Q2 = {q2}")
            return (q2_min + q1_max) / 2.0
        else:
            threshold = (q2_min + q1_max) / 2.0

    

if __name__ == "__main__":
    h = {
        10: 3,
        35: 5,
        45: 2,
        53: 3,
        92: 6,
        100: 5,
        115: 2,
        125: 3,
        128: 1,
        136: 6,
        164: 4,
        165: 2,
        166: 2,
        167: 2,
        192: 7,
        220: 7,
        255: 1
    }
    print(optimal_threshold(h))