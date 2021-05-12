import yaml
import os
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

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

    features = ["Area","GDP","Inflation","Life.expect","Military","Pop.growth","Unemployment"]
    name_and_features = ["Country"].extend(features)

    df = pd.read_csv(europe_csv, names=name_and_features)
    df = df[1:]
    df.index = range(0,len(df))

    X = df.loc[:, df.columns != "Country"]

    # print("DF:")
    # print(df)

    # X_new = []

    # for index, row in X.iterrows():
    #     arr = []
    #     # print(row)
    #     for c in X.columns:
    #         arr.append(row[c])
    #     X_new.append(arr)
        
    # X = X_new

    # print(X_new)

    # Standardizing the features
    X = StandardScaler().fit_transform(X)

    # print("X:")
    # print(X)

    n = len(features)

    pca = decomposition.PCA(n_components=n)
    pca.fit(X)
    X_pca = pca.transform(X)

    for i, component in enumerate(pca.components_):
        print("{} component: {}% of initial variance".format(i + 1, round(100 * pca.explained_variance_ratio_[i], 2)))
        print(" + ".join("%.3f x %s" % (value, name) for value, name in zip(component, features)))

    print(X_pca)

    # first_pca = X_pca[:,0]
    # new_first_pca = {
    #     'Country': [],
    #     'First Component': []
    # }

    # for index, row in df.iterrows():
    #     new_first_pca['Country'].append(row['Country'])
    #     new_first_pca['First Component'].append(first_pca[index-1])

    # df_pca = pd.DataFrame(new_first_pca, columns = ['Country', 'First Component'])

    df_pca = pd.DataFrame(X_pca, columns=["PC%d" % k for k in range(1,n + 1)])
    countries = df['Country'][0:]
    # print(countries)
    df_pca.insert(0, "Country", countries, True)

    first_pca = df_pca.iloc[:,:2]

    first_pca = first_pca.sort_values(by=['PC1'], ascending=True)

    print(first_pca)

    def myplot(score, coeff, coeffs, features_labels=None, records_labels=None):
        xs = score[:,0]
        ys = score[:,1]
        n = coeff.shape[0]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())

        scaled_x = xs * scalex
        scaled_y = ys * scaley

        plt.scatter(scaled_x, scaled_y, c = ['black'])
        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
            if features_labels is None:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
            else:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, features_labels[i], color = 'g', ha = 'center', va = 'center')

        for i in range(0, len(scaled_x)):
            if records_labels is None:
                plt.text(scaled_x[i]* 1.15, scaled_y[i]* 1.15, "Record"+str(i+1), color = 'r', ha = 'center', va = 'center')
            else:
                plt.text(scaled_x[i]* 1.15, scaled_y[i]* 1.15, records_labels[i], color = 'r', ha = 'center', va = 'center')

        plt.xlim(-0.75,0.75)
        plt.ylim(-0.75,0.75)
        plt.xlabel("PC%d %.2f%%" % (1, 100*coeffs[0]))
        plt.ylabel("PC%d %.2f%%" % (2, 100*coeffs[1]))
        plt.grid()

    #Call the function. Use only the 2 PCs.
    myplot(X_pca[:,0:2], np.transpose(pca.components_[0:2, :]), pca.explained_variance_ratio_[0:2], features, countries)
    # plt.show()