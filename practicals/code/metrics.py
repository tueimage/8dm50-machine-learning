def get_accuracy(gt, pred):
    """calculates accuracy of prediction compared to the ground truth
    
    parameters
    ----------
    gt : ground truth
    pred : predictions"""
    correct = 0
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            correct += 1
    return correct/len(gt) * 100

def get_mse(gt, pred):
    """calculates mean squared error of prediction compared to the ground truth
    
    parameters
    ----------
    gt : ground truth
    pred : predictions"""
    return ((gt - pred)**2).mean()