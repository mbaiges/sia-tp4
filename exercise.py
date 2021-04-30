import yaml
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

## data filenames
europe_csv = ""

config_filename = 'config.yaml'

with open(config_filename) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

    data_folder = config['data_folder']

    europe_csv = os.path.join(data_folder, config['europe_csv'])

# ver
# https://scikit-learn.org/stable/modules/decomposition.html
# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

if __name__ == "__main__":
    features = ["Country","Area","GDP","Inflation","Life.expect","Military","Pop.growth","Unemployment"]
    
    df = pd.read_csv(europe_csv, names=features)
    df = df[1:]

    # Standardizing the features
    x = StandardScaler().fit_transform(df)


    print(df)

