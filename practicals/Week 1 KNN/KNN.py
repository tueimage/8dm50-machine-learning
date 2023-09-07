# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

class KNN_LOO:
    """K nearest neighbors (NNs) algorithm, with leaving-one-out (LOO).
    
    Parameters
    ----------
    distances_test_to_training : 
        {datapoint_i: {datapoint_j: distance,
                       datapoint_k: distance,
                       datapoint_l: distance,
                       ...
                       }}
            datapoint_x : tuple(str, str, floats)
            distance : float
        Distance dict, with as keys each datapoint and as 
        values a dictionary with the distance of that datapoint to every
        other datapoint.
    
    k : int, default=5
        The amount of NNs that is to be considered. Default is 5.
    
    verbose : bool, default=False
        Print feedback to keep track of the progress of the algorithm.
        If False, the program will still print which class it predicted,
        the class is actually is part of, and if the algorithm thus
        correctly predicted the class.
    
    # visualize : bool, default=False
    #     Visualize the algorithm using the first 2 dimensions.
    
    
    Attributes
    ----------
    predicted_class : str
        The name of the predicted class.
        
    outcome : bool
        Whether the class was predicted correctly.
        
        
    Internal variables
    ------------------
    distances_test_to_training_sorted : dict{datapoint_i: dict{
                                                datapoints: distance}}
        Same as distances_test_to_training, but sorted by distance
        (values).
    
    NNs_dist_dict : {datapoint: dict{datapoints: float}
                     (datapoint : tuple(str, str, floats)) 
        Subset of distances_test_to_training of *k* datapoints that
        have the lowest distance to the test datapoint. Ordered by
        distance.
        
    NN_classes : list[strings] of len(NNs)
        The classes per NN.
        
    most_occurring_classes : list[strings]
        List containing the strings of the most occurring class names.
    
    
    Methods
    -------
    predict_class
        Determine nearest neighbor based on k neighbors (core of the
        algorithm with tie_handling).
    
    tie_handling
        Handle a tie in # of occurences of the classes of the NNs;
        choose the class whose neighbors are closer on average.
    
    outcome_eval
        Evaluate the outcome of the core of the algorithm (predict_class
        and tie_handling). 
    
    # visualize
    #     Visualize the dataset using its first two dimensions, with
    #     datapoints colored per class and the test datapoint the NNs
    #     highlighted.
    """
    def __init__(self, distances_test_to_training, k=None, verbose=False, 
                 # visualize=False
                 ):
        
        #### Assign instance variables, and check the input
        if not isinstance(distances_test_to_training, dict):
                raise TypeError('distances_test_to_training should be a nested'
                                ' dict.')
        else:
            self.distances_test_to_training = distances_test_to_training
        
        if k == None:  # Default k
            self.k = round((len(distances_test_to_training[tuple(
                distances_test_to_training)[0]]))**0.5) + 1   
                # sqrt(# of datapoints), the 1 is because the test
                # datapoint is left out
        elif not isinstance(k, int):
            raise TypeError('k should be an int (>0).')
        elif k < 1:
            raise ValueError('k has to be an integer higher than 0.')
        else:
            self.k = k
        
        if not isinstance(verbose, bool):
            raise TypeError('verbose should be a bool.')
        else:
            self.verbose = verbose
            
        # self.visualize = visualize
        
        #### Run algorithm
        self.predict_class()
        
        #### Evaluate performance
        self.outcome_eval()
        
    
    def predict_class(self):
        """Determine nearest neighbor based on k neighbors."""
        #### Determine the *k* closest (training) datapoints to the test
        #### datapoint (i.e. the *k* lowest values in the dict)
        if self.verbose: print('Determining the *k* closest (training) '
                               'datapoints to the test datapoint...',
                               end=' ')
        NNs_dist_dict = dict(  # Convert the NNs with distances back to dict
            sorted(
                list(self.distances_test_to_training.items()),  
                    # ^sorted() does not work with dicts
                key=lambda x: x[1])  # Sort by the distance i.e. the values
            [:self.k]   # Take only the first *k* closest datapoints
            )
        if self.verbose: print('Done!')
        
        #### Determine occurences of the classes in the NNs
        # Determine the classes of the nearest neighbors
        NN_classes = [datapoint_tuple[1] 
                      for datapoint_tuple, dist in NNs_dist_dict.items()]
            # datapoint_tuple[1] = class (str)
            # NN_classes : list[strings] (Ordered by distance to test
            #                             datapoint.)
        
        # Find occurences of the classes in the NNs
        occurences_NN_classes = dict((class_i, NN_classes.count(class_i)) 
                                     for class_i in NN_classes)
        if self.verbose: print('Occurences of the classes in the NNs are: ',
                               occurences_NN_classes)
        
        
        #### Determine the most occurent class(es) out of the nearest
        #### neighbors
        if self.verbose: print('Determining the most occurent classes of the '
                               'NNs...', end=' ')
        most_occurring_classes = [
            k for k in occurences_NN_classes 
            if occurences_NN_classes[k] == max(occurences_NN_classes.values())
            ]
        if self.verbose: print('Done!')
        
        #### Handle ties: check if number of occurences of >1 classes is
        #### the highest i.e. there is a tie. If yes: choose class whose
        #### neighbors are closer on average.
        # If there is a tie:
        if len(most_occurring_classes) > 1:  
            self.tie_handling(most_occurring_classes, NNs_dist_dict)
            if self.verbose: 
                print(f'Predicted class: {self.predicted_class}')
        # If there is no tie:
        else:  
            self.predicted_class = most_occurring_classes[0]
            if self.verbose: 
                print(f'Predicted class: {self.predicted_class}')
            
    
    def tie_handling(self, most_occurring_classes, NNs_dist_dict):
        """Handle a tie in # of occurences of the classes of the NNs.
        Choose the class whose neighbors are closer on average.
        
        Parameters
        ----------
        most_occurring_classes : list[strings]
            List containing the strings of the class names.
            
        NNs_dist_dict : {datapoint: dict{datapoints: float}
                         (datapoint : tuple(str, str, floats)) 
            Subset of distances_test_to_training of *k* datapoints that
            have the lowest distance to the test datapoint. Ordered by
            distance.
            
        Internal variables
        ------------------
        _most_occurring_class_w_dist : dict{'cls1': dist_tot_cls_1, ...}
            Dict with the class names (str) as keys, and the sum of the
            distances between all datapoints belonging to that class, 
            and the test datapoint.
            
        Returns
        -------
        self.predicted_class : str
            The class that the test datapoint is predicted to have = the
            class whose neighbors are closer on average.
        """
        if self.verbose: print('There is a tie; determining closest class out '
                               'of:', *most_occurring_classes)
        
        if self.k == 1:
            raise Exception('k=1 so tie is not possible. Something is wrong '
                            'in the code.')
        
        #### Out of the NNs that are of the most occuring classes,
        #### determine the distance per most occurring class.
        # Allocate
        _most_occurring_classes_w_dist = {
            i_class: 0. for i_class in most_occurring_classes}
        # Find total distance per most occurring class 
        for datapoint_i, dist_i in NNs_dist_dict.items():
            class_of_NN_i = datapoint_i[1]
            if class_of_NN_i in most_occurring_classes:
                _most_occurring_classes_w_dist[class_of_NN_i] += dist_i
        # Find class with the lowest total distance = predicted class
        self.predicted_class = min(_most_occurring_classes_w_dist.values())
        
        if self.verbose: print('Finished determining closest class i.e. '
                               'handling the tie.')
    
    
    def outcome_eval(self):
        """Determine whether the algorithm predicted the correct class
        or not.
        
        Returns
        -------
        self.outcome : bool
            True = correctly predicted.
        """
        if self.verbose: print('Evaluating performance of KNN...', end=' ')
        
        # To save computation time:
        _test_datapoint = next(iter(self.distances_test_to_training))  # 1st
            # key of the dict
        _true_test_datapoint_class = _test_datapoint[1]  # class on idx 1.
        
        # Outcome evaluation:
        self.outcome = (self.predicted_class == _true_test_datapoint_class)
        
        # Return and print the outcome:
        if self.outcome:
            print('The class of the test datapoint has been correctly '
                  f'predicted ({self.predicted_class}).\n')
        else: 
            print('The class of the test datapoint has NOT been correctly '
                  'predicted: \n'
                  f'predicted class = {self.predicted_class} \n'
                  f'actual class = {_true_test_datapoint_class}.\n')
        
        return self.outcome
        
    
    # def visualize(self):
    #     """Visualize the result of the algorithm using the first two
    #     dimensions."""
    #     plt.figure()
        
    #     classes_list = [
    #         datapoint[1] for datapoint in self.distances_test_to_training]
        
    #     x = [x[0] for x in self.distances_test_to_training]  # dim 1
    #     y = [x[1] for x in self.distances_test_to_training]  # dim 2
        
    #     plt.scatter(x, y, label = classes_list, c = classes_list)
    #     plt.scatter(self.test_datapoint[2], self.test_datapoint[3], 
    #                 linewidths=4.5)
        
    #     title = f'KNN with k={self.k}'
    #     plt.title(title)
    #     plt.xlabel('Feature 1')
    #     plt.ylabel('Feature 2')
