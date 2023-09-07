from KNN import KNN_LOO
from utils import read_data, standardize, distance_all_datapoints
import time
import distance_calc
import matplotlib.pyplot as plt


#%% PARAMETER CHOICES for KNN-LOO.
distance_method = distance_calc.euclidean_squared
# distance_method = distance_calc.manhattan
verbose = True  # For more print statements.
# verbose = False  # For less print statements.
# k_list = [1, 5, 10, 20, 50]
# k_list = [1, *(range(5,51,5))]  # 1 5 10 15 ... 45 50
k_list = [1,2,3,4,5]#[1, *(range(5,999,5))]  # 1 5 10 15 ... 45 50
# k_list = [50]
# k_list = range(1, 51)
# k_list = [99]
# k_list = range(1001, 5)  # n_of_datapoints-1, -1 for the test datapoint
# k_list = [1, 15]

# filename = 'QSAR_reduced_40.csv'   # nelem=3200  (n=100, p=32)
filename = 'QSAR_2.csv'   # nelem=41.6k  (n=1300, p=32)


# %%
##### READ DATA ########################################################
print('Reading data...', end=' ')
data = read_data(filename)  # List[lists], no headers
print('Done!')

# For later use:
n_of_datapoints = len(data)
n_of_features = len(data[0]) - 2  # Exclude name and class.
nelem = n_of_datapoints * n_of_features

##### STANDARDIZE DATA #################################################
print('Standardizing data...', end=' ')
# Delete the name and class (strings); they cannot be standardized:
names = [datapoint.pop(0) for datapoint in data]
classes = [datapoint.pop(0) for datapoint in data]

# Standardize :
data = standardize(data)

# Add back the name and class:
for idx, datapoint in enumerate(data):
    datapoint.insert(0, classes[idx])
for idx, datapoint in enumerate(data):
    datapoint.insert(0, names[idx])
print('Done!')

#%% CALCULATE DISTANCES
print('Calculating distances... (this will take approx. '
      f'{4.65/10000 * nelem:.0f} s)', end=' ')
start_time = time.time()
# Covert list of lists -> list of tuples
data_list_of_tuples = [tuple(internal_list) for internal_list in data]

# Calculate distances between all datapoints
distance_dict = distance_all_datapoints(data_list_of_tuples, distance_method)
end_time = time.time()
computation_time_dists = end_time - start_time
print('Done!')
#%% QUESTION 2
### ----------
### Determine for at least 5 different values of k what the error score
### of your k-nearest neighbour algorithm is, i.e., the number of times 
### that the algorithm generates a different label than the real label 
### when run for all 100 data points with the leave-one-out procedure.

error_score_dict = {}
error_score_dict_for_plot = {}
computation_time_dict = {}

for k in k_list:
    print(f'k = {k}\n')
    
    start_time = time.time()
    
    KNN_outcome_list = []  # To keep track of the outcomes during the loop
    
    # Run for every single datapoint as test datapoint
    for idx, datapoint_i in enumerate(data_list_of_tuples):
        distances_test_to_training = distance_dict[datapoint_i]
        n_of_digits = len(str(n_of_datapoints))
        print(f'-- Datapoint {idx+1:0{n_of_digits}d} of {n_of_datapoints} (k={k}) --')
        
        # Run KNN and save the outcome: True if correct
        KNN_outcome_list.append(
            KNN_LOO(distances_test_to_training, k, verbose).outcome
            )
        
    # Computation time check
    end_time = time.time()
    computation_time = end_time - start_time
    print('\n')
    print('==================================================================')
    print(f'For {n_of_datapoints} datapoints with {len(datapoint_i)} '
          f'features, k={k}, with distance method = {distance_method}, '
          f'it takes {computation_time:.3f} s to run the KNN algorithm and '
          'compute the error scores.')
    # Determine error score i.e. number of times that the algorithm
    # generates a different label than the real label, and save it 
    # for the current *k*.
    error_score = KNN_outcome_list.count(False)  # int: 0-100
    error_score_proportion = str(error_score) + '/' + str(n_of_datapoints)
    error_score_dict[k] = error_score_proportion
    error_score_dict_for_plot[k] = error_score
    computation_time_dict[k] = computation_time
    print('==================================================================')
    print('Error scores per k:')
    # print(error_score_dict)
    print('\n'.join(f'{key}: {value}' for key, value in 
                    error_score_dict.items()))
    print('==================================================================')
    print('Computation time per k in s:')
    # print(computation_time_dict)
    print('\n'.join(f'{key}: {value:.2f}' for key, value in 
                    computation_time_dict.items()))

# Print computation time distances
print('==================================================================')
print(f'Calculate dists time: {computation_time_dists:.02f} s')
print('Calculate dists time per datapoint: '
      f'{computation_time_dists/n_of_datapoints:.05f} s')

# Plot: k vs error score
x = list(error_score_dict_for_plot.keys())
y = list(error_score_dict_for_plot.values())
plt.scatter(x, y)
plt.plot(x, y, '--')
plt.title(f'Error score as a function of k (n={n_of_datapoints})')
plt.xlabel('k')
plt.ylabel(f'Error score (n = {n_of_datapoints})')
plt.show()

# Plot: k vs computation time
y = list(computation_time_dict.values())
plt.scatter(x, y)
plt.plot(x, y, '--')
plt.suptitle(f'Computation time as a function of *k* (# elem={nelem})')
plt.title(f'(computing distances took {computation_time_dists:.2f} s)')
plt.xlabel('k')
plt.ylabel('Computation time [s]')
plt.show()




