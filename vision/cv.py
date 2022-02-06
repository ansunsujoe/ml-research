import numpy as np

def create_hist(array):
    """
    Create Histogram from 2D Numpy Array
    """
    # Histogram as a dictionary
    hist = {}
    rows, cols = array.shape
    
    # Iterate through the array
    for r in range(rows):
        for c in range(cols):
            if hist.get(array[r][c]) is None:
                hist[array[r][c]] = 1
            else:
                hist[array[r][c]] += 1
    return hist

def linear_scaling(array, A=0, B=255):
    # Identify range of current histogram
    hist = create_hist(array)
    a = min(hist.keys())
    b = max(hist.keys())
    
    # Create a new histogram that has the scaling mapping
    mapping = {}
    for x in hist:
        # Formula for linear scaling
        new_x = ((x - a) * (B - A) / (b - a)) + A
        mapping[x] = round(new_x)
        
    # Apply the scaling to the array
    scaled = np.zeros(array.shape)
    for r in range(scaled.shape[0]):
        for c in range(scaled.shape[1]):
            scaled[r][c] = mapping[array[r][c]]
    
    # Return
    return scaled

def hist_equalization(array):
    # Define initial variables
    k = np.max(array)
    

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
            print(f"Total Error: {qerror(hist, threshold)}")
            return (q2_min + q1_max) / 2.0
        else:
            threshold = (q2_min + q1_max) / 2.0

    

if __name__ == "__main__":
    h = {
        6: 6,
        10: 2,
        17: 7,
        88: 1
    }
    print(optimal_threshold(h))