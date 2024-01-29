import numpy as np

def topsis(data, weights, impacts):
    # Normalize the decision matrix
    norm_data = data / np.linalg.norm(data, axis=0)

    # Multiply each column by its weight
    weighted_data = norm_data * weights

    # Find the ideal and negative-ideal solutions
    ideal_best = np.max(weighted_data, axis=0)
    ideal_worst = np.min(weighted_data, axis=0)

    # Calculate the Euclidean distances to the ideal and negative-ideal solutions
    dist_best = np.linalg.norm(weighted_data - ideal_best, axis=1)
    dist_worst = np.linalg.norm(weighted_data - ideal_worst, axis=1)

    # Calculate the relative closeness to the ideal solution
    relative_closeness = dist_worst / (dist_best + dist_worst)

    # Apply impact on the relative closeness
    topsis_score = relative_closeness
    for i in range(len(impacts)):
        if impacts[i] == '-':
            topsis_score[i] = 1 / (relative_closeness[i] + 0.0001)

    return topsis_score

