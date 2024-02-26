# This is the code to implement K-Nearest Neighbors algorithm
# on Tasls.pkl
# The original code can be found: https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/KNN/KNN.ipynb



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# For scaling data
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn import metrics
import pickle


#  Step 1: Load Data
with open('PCA_feature_5_dim_3.pkl', 'rb') as f:
    df = pickle.load(f)

print(df)


#  Arrange Data into Features Matrix and Target Vector

# X = df.loc[:, ['principal component 1', 'principal component 2']]
X = df.loc[:, df.columns != 'target']
X.shape


df.loc[df['target'] == 'easy'] = 0
df.loc[df['target'] == 'difficult'] = 1
y = df.loc[:, 'target'].values
y=y.astype('int')
y.shape

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = .2)


#  Step 2: KNN in scikit-learn

# Step 2.1: Import the model you want to use, In sklearn, all machine learning models are implemented as Python classes

from sklearn.neighbors import KNeighborsClassifier

# Step 2.2: Make an instance of the Model

knn = KNeighborsClassifier(n_neighbors=5)
print(knn)

# Step 2.3: Train the model on the data, storing the information learned from the data. Model is learning the relationship between features and labels

knn.fit(X_train, y_train)

# Step 2.4: Predict the labels of new data, Uses the information the model learned during the model training process

predictions = knn.predict(X_test)

predictions


# Step 2.5: calculate classification accuracy

score = knn.score(X_test, y_test)
score


#  Step 3: Visualizing Data

cmap_light = ListedColormap(['orange', 'cyan']) #, 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c'])#, 'darkblue'])
h = .02  # step size in the mesh


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_train.loc[:, 'principal component 1'].values.min() - 1, X_train.loc[:, 'principal component 1'].values.max() + 1
y_min, y_max = X_train.loc[:, 'principal component 2'].values.min() - 1, X_train.loc[:, 'principal component 2'].values.max() + 1
# z_min, z_max = X_train.loc[:, 'principal component 3'].values.min() - 1, X_train.loc[:, 'principal component 3'].values.max() + 1


xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
                         # np.arange(z_min, z_max))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])  #, zz.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='nearest')

# Plot also the training points
plt.scatter(X_train.loc[:, 'principal component 1'].values,
            X_train.loc[:, 'principal component 2'].values,
            c=y_train,
            cmap=cmap_bold,
            edgecolor='k',
            s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Class classification (k = 5). 2 components")


d = 1