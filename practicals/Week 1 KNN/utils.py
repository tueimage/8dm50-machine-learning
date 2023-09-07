
#%%
### Load the data
def read_data(filename):
    """Read datafile in the current directory and return as list of
    lists, without headers.
    
    Parameters
    ----------
    filename : str
        The filename of the dataset, should be of .csv filetype.

    Returns
    -------
    list[tuples]
    """
    import pandas as pd
    
    # To find the file in the current directory:
    if not filename.startswith('./'):
        filename = './' + filename
    
    # Read file
    df = pd.read_csv(filename, header=None)
    
    #### Preprocessing
    # Convert to list
    data_list_of_lists = df.values.tolist()
    
    # Remove headers
    data_list_of_lists_no_headers_w_strings = data_list_of_lists[1:]
    
    # Convert str(float) in data to float:
    data_list_of_lists_no_headers = [
        [*row[:2], *map(float, row[2:])] 
        for row in data_list_of_lists_no_headers_w_strings
        ]
    # for idx_datapoint, datapoint in enumerate(data_list_of_lists_no_headers):
    #     for idx_feature, feature_val in enumerate(datapoint):
    #         if idx_feature not in (0, 1):
    #             data_list_of_lists_no_headers[idx_datapoint][idx_feature] = float(feature_val)
    
    # for i, datapoint in enumerate(data_list_of_lists_no_headers):
    #     for i2, v in enumerate(list):
    #         try:
    #             data_list_of_lists_no_headers[i][i2] = float(v)
    #         except ValueError:  # If the element is a str with letters
    #             pass
        
    return data_list_of_lists_no_headers

# print(read_data(filename)[0])

#%%
def cumu_mov_mean(x=[0,1,2], lst = False):
    """The cumulative moving average is calculated with the following formula:
    C[n+1] = C[n] + (x[n] - C[n])/(n+1)
    
    Parameters 
    ------------
        x : list 
            x is a list from which the cumulative moving average will be calculated
        lst: Boolean 
            lst is a Boolean which can be set to true when complete list of cumulative averages is wanted. 
                Otherwise only the last value, so the final average (mean over entire list) is returned. 
    Returns
    ------------
        cumulative_list : list
            cumulative_list is a list with all the calculated cumulative moving averages 
        cumulative_list[-1] : float
            cumulative_list[-1] is the final cumulative moving average, so the last value of the complete list and the 
                mean of the entire list. 
    """ 
    cumulative_list = [0] * (len(x)+1)  #make list of zeros with same length as the input
    cumulative_list[0] = 0        #set first value to zero
       
    for i in range(len(x)):
        cumulative_list[i+1] = cumulative_list[i]+((x[i]-cumulative_list[i])/(i+1))
    
    if lst == False:
        return cumulative_list[-1]
    elif lst == True:
        return cumulative_list
    
def standardize(lst, method='sklearn'):    
    """Scale (standardize) a given list. 
    - method == 'sklearn'  (default):
        Uses sklearn.preprocessing.StandardScaler().fit_transform.
    - method == 'custom':
        Uses the following calculation: xij' = (xij - mean(xj))/s,
        where the mean is calculated as a cumulative moving average.
    
    Parameters 
    ----------
    lst : list 
        List with floats / integers.
    
    Returns
    -------
    stand_lst : list
        List containing standardized values of given input list.
    """
    if method == 'sklearn':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = scaler.fit_transform(lst)
        stand_lst = data.tolist()
        return stand_lst
    
    elif method == 'custom':
        # Calculate average of the entire list using cumu_mov_mean 
        mean = cumu_mov_mean(x=lst, lst=False)
        
        if mean == 0:  
            # Otherwise divide by zero for std and resulting in NaN
            # after standardization mean should be 0, so if it's already
            # 0, the values are already standardized 
            return lst 
        
        # Prepare variables for calculation of standardized values 
        N = len(lst)
        sigma = 0  # To store variance later 
        stand_lst = [0]*N  # Intialize output
        
        # Calculation of variance 
        for i in range(N):
            sigma += (lst[i] - mean)**2
        sigma = 1 / (N-1) * sigma 
        
        std = sigma**(1/2) # Calculation of standard deviation 
        
        # Standardize every value in lst 
        for i in range(N):
            stand_lst[i] = (lst[i] - mean)/std
            
        return stand_lst
    
def get_label_names(data, column_w_labels):
    """Get the unique labels (or classes) from a DataFrame.
    
    Parameters
    ----------
    data : DataFrame
    column_w_labels : int
        The index of the column which contains the labels.
    
    Returns
    -------
    label_names_list : list[str]
        A list containing the labels (str) of the dataset.
    """
    label_names_list = data.iloc[:, column_w_labels].unique()
    return label_names_list
    
    
def distance_all_datapoints(data_list_of_tuples, distance_method):
    """Calculate the distance between each datapoint and all other
    datapoints. Return as dictionary, with as keys each datapoint and as values a
    dictionary with the distance 
    of that datapoint to every other datapoint.
    
    Parameters
    ----------
    data_list_of_tuples : list[tuples(str, str, floats)]
        
    distance_method : str
        The distance method to be used for calculating distances between
        datapoints.
    
    Returns
    -------
    distance_dict : dict{datapoint: dict{datapoint: floats}}
                        (datapoint : tuple(str, str, floats))
        Distance dict, with as keys each datapoint and as 
        values a dictionary with the distance of that datapoint to every
        other datapoint.
    """
    if not isinstance(data_list_of_tuples, list):
        raise TypeError('data_list_of_tuples should be a list of tuples with '
                        '(str, str, floats).')
    from inspect import isfunction
    if not isfunction(distance_method): 
        raise TypeError('distance_method should be a function/method, '
                            'e.g. distance_calc.euclidean_squared')
        
    distance_dict = {}
    
    for dp_i in data_list_of_tuples:

        # The internal dict depicting distances to datapoint_i:
        distances_dp_i = {}  
        
        for dp_j in data_list_of_tuples:
            
            # If dp_j is already a key in dist_dict, its distances
            # to other datapoints have already been calculated, so
            # instead of again calculating the distance between
            # dp_i and dp_j the distance between dp_j
            # and dp_i is re-used.
            if dp_j in distance_dict.keys(): 
                distances_dp_i[dp_j] = distance_dict[dp_j][dp_i]
            
            # The distance between a datapoint and itself does not need
            # to be calculated. For any other datapoint, the distance is
            # determined.
            elif dp_j != dp_i: 
                distances_dp_i[dp_j] = distance_method(
                    list(dp_i[2:]), list(dp_j[2:])) 
                    
        # For the key a in the dictionary distance_dict, the value is
        # the dictionary of distances to every other datapoint, as
        # calculated above.
        distance_dict[dp_i] = distances_dp_i 

    return distance_dict