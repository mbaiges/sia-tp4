import yaml
import os
import pandas as pd
import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt

## data filenames
europe_csv = ""

config_filename = 'config.yaml'

class Neuron:

    def __init__(self, dataset, use_dataset = False):
        if use_dataset:
            self.init_weights(dataset)
        else:
            self.init_weights_random(dataset.shape[1])
        self.hit_count = 0

    def init_weights(self, dataset):
        idx = np.random.randint(0, dataset.shape[0])
        entry = dataset[idx,:]
        self.weights = copy.copy(entry)

    def init_weights_random(self, length):
        np.random.uniform(low=-1, high=1, size=(length,))


with open(config_filename) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

    data_folder = config['data_folder']

    europe_csv = os.path.join(data_folder, config['europe_csv'])

def get_data():
    features = ["Area","GDP","Inflation","Life.expect","Military","Pop.growth","Unemployment"]
    name_and_features = ["Country"].extend(features)

    df = pd.read_csv(europe_csv, names=name_and_features)
    df = df[1:]
    df.index = range(0,len(df))

    return df

def standardize_data(data):
    return StandardScaler().fit_transform(data)

def init_output_neuron_matrix(k, entries, init_with_dataset):
	output_neuron_mtx = []
	for i in range(0,k):
		output_neuron_mtx.append([])
		for j in range(0, k):
			neuron = Neuron(entries, init_with_dataset)
			output_neuron_mtx[i].append(neuron)
	return output_neuron_mtx

def get_distance(arr1, arr2):
    dist = 0
    for i in range(0, arr1.shape[0]):
        dist += (arr1[i] - arr2[i])**2
    return math.sqrt(dist)

def find_neuron(output_neuron_mtx, entry):
    min_i = 0
    min_j = 0
    min_dist = math.inf
    for i in range(0, len(output_neuron_mtx)):
        for j in range(0, len(output_neuron_mtx[0])):
            d = get_distance(output_neuron_mtx[i][j].weights, entry)
            if d < min_dist:
                min_i = i
                min_j = j
                min_dist = d
    return (min_i, min_j)          

def update_neuron(output_neuron_mtx, pos, entry, eta_f, it):
    (i, j) = pos
    neuron = output_neuron_mtx[i][j]
    eta = eta_f(it)
    neuron.weights += eta * (entry - neuron.weights)
    

def get_neighbours(output_neuron_mtx, idx, radius):
    #print(f"RADIUS: {radius}")
    output_mtx_row_limit = len(output_neuron_mtx)
    row_bottom_limit = idx[0] - radius if idx[0] - radius >= 0 else 0
    #print(f"ROW_BOTTOM_LIMITS: {row_bottom_limit}")
    row_bottom_limit = int(row_bottom_limit)
    row_upper_limit = idx[0] + radius if idx[0] + radius < output_mtx_row_limit else output_mtx_row_limit
    #print(f"ROW_UPPER_LIMITS: {row_upper_limit}")
    row_upper_limit = int(row_upper_limit)

    output_mtx_col_limit = len(output_neuron_mtx[0])
    col_bottom_limit = idx[1] - radius if idx[1] - radius >= 0 else 0
    col_bottom_limit = int(col_bottom_limit)
    col_upper_limit = idx[1] + radius if idx[1] + radius < output_mtx_col_limit else output_mtx_col_limit
    col_upper_limit = int(col_upper_limit)

    #print(f"LIMITS: ({row_bottom_limit},{row_upper_limit}), ({col_bottom_limit},{col_upper_limit})")

    ne = []
    for i in range(row_bottom_limit, row_upper_limit):
        for j in range(col_bottom_limit, col_upper_limit):
            dist = math.sqrt((idx[0] - i)**2 + (idx[1] - j)**2)
            # print(dist)
            if dist <= radius:
                ne.append((i,j))
    return ne

def update_neighbours(output_neuron_mtx, best_match_idx, entry, radius, eta_f, it):
    
    ne = get_neighbours(output_neuron_mtx, best_match_idx, radius)
    # print(ne)
    # update neuron
    for (i, j) in ne:
        update_neuron(output_neuron_mtx, (i, j), entry, eta_f, it)
    
def process_input(output_neuron_mtx, entry, radius, eta_f, it):
	(best_i, best_j) = find_neuron(output_neuron_mtx, entry)  
	# print(f"ENTRY: {entry}")
	#print(f"BEST: ({best_i}, {best_j})")  
	output_neuron_mtx[best_i][best_j].hit_count += 1
	update_neighbours(output_neuron_mtx, (best_i, best_j), entry, radius, eta_f, it)
	return 0

def display_results(output_neuron_mtx):
	k = len(output_neuron_mtx)
	a = np.zeros((k, k))
	for i in range(0,k):
		for j in range (0,k):
			a[i,j] = output_neuron_mtx[i][j].hit_count
	
	plt.imshow(a, cmap='hot', interpolation='nearest')
	plt.show()

def display_u_matrix(output_neuron_mtx, radius): #grey matrix
	k = len(output_neuron_mtx)
	a = np.zeros((k, k))
	for i in range(0,k):
		for j in range (0,k):
			ne = get_neighbours(output_neuron_mtx, (i,j), radius)
			#print(ne)
			d = 0.0
			cant = len(ne)
			for (ne_i, ne_j) in ne:
				d += get_distance(output_neuron_mtx[i][j].weights,output_neuron_mtx[ne_i][ne_j].weights)
			a[i,j] = d/cant
	plt.imshow(a, cmap='hot', interpolation='nearest')
	plt.show()


def display_final_assignments(data, std_data, output_neuron_mtx):
    k = len(output_neuron_mtx)
    names = [ [ [] for j in range(0, k) ] for i in range(0, k) ]
    a = np.zeros((k, k))
    # print(data)
    for i in range(0, std_data.shape[0]):
        entry = std_data[i]
        (x,y) = find_neuron(output_neuron_mtx, entry)
        a[x,y] += 1
        names[x][y].append(data['Country'].values.tolist()[i])
    a = a / np.std(a)
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.show()

def kohonen(entries, k, initial_radius, init_with_dataset, eta_f):
    epochs = 25 * entries.shape[0]
    radius = initial_radius
    # print(f"RADIUS: {radius}")

    # initialize weights (k * input)
    output_neuron_mtx = init_output_neuron_matrix(k, entries, init_with_dataset)

    # iterar por todas las entradas y por cada entrada llamar a process_input
    for epoch in range(0,epochs):
        aux_entries = copy.copy(entries)
        random.shuffle(aux_entries)
        for i in range(0, aux_entries.shape[0]):
            entry = aux_entries[i, :]
            #print(entry)		
            process_input(output_neuron_mtx, entry, radius, eta_f, epoch+1)
        if radius - 1 > 1:
            radius -= 1 #TODO: Lo podemos complejizar despues para que sea mas adaptativo el radio
        
    display_results(output_neuron_mtx)
    display_u_matrix(output_neuron_mtx, radius)

    return output_neuron_mtx
    
if __name__ == '__main__':
    
    # get_data
    df = get_data()

    data = df.loc[:, df.columns != "Country"]
    data = np.asarray(data.values.tolist())

    # standarize
    std_data = standardize_data(data)

    # kohonen
    k = 16
    radius = math.sqrt(2)
    init_with_dataset = True
    
    def eta_f(i):
        return 1.0/i

    output_neuron_mtx = kohonen(std_data, k, radius, init_with_dataset, eta_f)

    display_final_assignments(df, std_data, output_neuron_mtx)

    exit(0)