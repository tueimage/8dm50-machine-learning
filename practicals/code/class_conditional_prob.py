from scipy.stats import norm

# Function to compute class-conditional probabilities
def class_conditional_prob(X, y, feature_idx):
    classes = np.unique(y)
    means = []
    stds = []
    for c in classes:
        X_c = X[y == c, feature_idx]
        means.append(np.mean(X_c))
        stds.append(np.std(X_c))
    return means, stds