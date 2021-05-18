import yaml
import os
import pandas as pd
import numpy as np
import random
import math
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

## data filenames
europe_csv = ""

config_filename = 'config.yaml'

with open(config_filename) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

    data_folder = config['data_folder']

    europe_csv = os.path.join(data_folder, config['europe_csv'])

def get_data():
    features = ["Area","GDP","Inflation","Life.expect","Military","Pop.growth","Unemployment"]
    name_and_features = ["Country"].extend(features)

    df = pd.read_csv(europe_csv, names=name_and_features)

    return df
        
def standardize_data(data):
    return StandardScaler().fit_transform(data)

class Neuron:

    def __init__(self, weights):
        self.weights = weights

    def apply_correction(self, learning_level, result, entry):
        s = result
        # print("----------------- COMPONENTS -----------------")
        # print(f'self.weights = {self.weights}')
        # print(f'entry = {entry}')
        # print(f'(entry - s*self.weights) = {(entry - s*self.weights)}')
        # print(f's * (entry - s*self.weights) = {s * (entry - s*self.weights)}')
        # print(f'learning_level * s * (entry - s*self.weights) = {learning_level * s * (entry - s*self.weights)}')
        # print("----------------------------------------------")

        # self.weights = self.weights + learning_level * s * (entry - s*self.weights) # filminas
        self.weights = self.weights + learning_level * s * (entry - self.weights*s) # https://www.cse-lab.ethz.ch/wp-content/uploads/2019/10/tutorial_3_ojas_rule_pdf.pdf 

        # w = self.weights
        # norma = math.sqrt(sum(w*w))
        # w = w / norma

        # self.weights = w

    def evaluate(self, entry):
        excitation = np.inner(entry, self.weights)
        activation = excitation #es lineal
        return activation

def oja(data, epochs, initial_learning_level):
    learning_level = initial_learning_level
    X = data

    # init neuron

    initial_weights = np.random.uniform(-1, 1, X.shape[1])

    neuron = Neuron(initial_weights)
        
    # train

    #create random= index array 
    orders = [a for a in range(0, X.shape[0])]
    
    epoch_n = 0

    while epoch_n < epochs:
        random.shuffle(orders)
        
        # print(f'({epoch_n}) Weights: {neuron.weights}')

        i = 0        
        while i < len(orders):
            
            #access index from order array
            indx = orders[i]
            pos_X = X[indx, :]

            #evaluate chosen input
            activation = neuron.evaluate(pos_X)
            neuron.apply_correction(learning_level, activation, pos_X)

            i += 1

        epoch_n += 1


        if learning_level / 2 > 0.0001:
            learning_level = learning_level / 2 

    w = neuron.weights
    norma = math.sqrt(sum(w*w))
    w = w / norma
        
    print("Finished training")
    s = 'Weights: ['

    for weight in w:
        s += f'{weight:.3f} '

    s += ']'
    print(s)

    results = w

    return results

    
if __name__ == '__main__':
    
    # get_data
    df = get_data()

    data = df.loc[:, df.columns != "Country"]
    data = np.asarray(data.values.tolist())

    # standarize
    std_data = standardize_data(data)

    # oja
    epochs = 15000
    initial_learning_level = 0.1

    weights = oja(std_data, epochs, initial_learning_level)
    
    exit(0)
