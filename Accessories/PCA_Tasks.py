# This is the code to implement PCA algorithm from https://builtin.com/machine-learning/pca-in-python
# on Tasls.pkl
# The code can be found: https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Data_Visualization_Iris_Dataset_Blog.ipynb


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle


# Step 1: Load Tasks.pkl and create pandas using features

with open('Tasks.pkl', 'rb') as f:
    Tasks0 = pickle.load(f)

# Split and obtain easy and difficult tasks

# Obtain features
# task.init_policy
# task.plain_feature
# task.feature
# 记录pg-rl和TaDeLL的values，policies，gradients的记录
# task.values_pg = values_pg
# task.values_TaDeLL = values_TaDeLL
# task.policy_pg =policies_pg
# task.policy_TaDeLL = policies_TaDeLL
# task.gradients_pg = gradients_pg
# task.gradients_TaDeLL = gradients_TaDeLL
# 一些人工特征提取
# task.cost_gap = values_TaDeLL[0] - values_pg[0]
# task.first_gradient =
# task.distance = np.linalg.norm(task.feature - origin)
# task.cos_distance = 1 - np.dot(np.squeeze(task.feature), np.squeeze(origin_cos)) / (np.linalg.norm(np.squeeze(task.feature)) * np.linalg.norm(np.squeeze(origin_cos)))
# 下面的计算方法可以得到一样的结果
# task.cos_distance = pdist(np.vstack([np.squeeze(task.feature), np.squeeze(origin_cos)]), 'cosine')


index = [[], []]
Tasks = [[], []]
type = ['easy', 'difficult']

g = [11, 12, 13, 16, 25]  # The ones in easy but not good
index[0] = list(set(list(range(0, 41))) - set(g))
index[1] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 20, 21, 22, 23, 24, 26, 28, 29, 33, 34, 36]  # Index for difficult tasks

data = []
for x in range(2):
    # Iterate over easy and difficult tasks separately
    Tasks[x] = [Tasks0[x][y] for y in index[x]]  # Pick easy or difficult tasks

    for task in Tasks[x]:
        # a = list(task.init_policy['theta'].flatten())  \
        #     + list(task.plain_feature.flatten()) \
        #     + list(task.policy_pg[-1].flatten()) + [task.values_pg[0]] + [task.values_pg[-1]] \
        #     + list(task.gradients_pg[0].flatten()) \
        #     + [task.cos_distance] + [task.distance] \
        #     + [type[x]]
        a = list(task.plain_feature.flatten()) \
            + [type[x]]
        # Initial policy, 2, feature, 5
            # Optimal policy, 2, initial value, 1, optimal value, 2
            # initial gradient, 2
            # cos distance, 1, distance, 1
            # Type
        data.append(a)


# col = ['policy1', 'policy2',
col = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5',
       # 'optpolicy1', 'optpolicy2', 'value1', 'value2',
       # 'inigradient1', 'inigradient2',
       # 'cosdistance', 'distance',
       'target']
df = pd.DataFrame(data = np.array(data), index = range(0, len(data)), columns = col)



# Step 2: Standardize the Data

# features = ['policy1', 'policy2', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5','optpolicy1', 'optpolicy2', 'value1', 'value2',]
features = col[:-1]

x = df.loc[:, features].values

y = df.loc[:,['target']].values

x = StandardScaler().fit_transform(x)

pd.DataFrame(data = x, columns = features).head()


# Step 3: PCA Projection to 2D or 3D or 4D

dim = 3

pca = PCA(n_components=dim)

principalComponents = pca.fit_transform(x)


col_name = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6']
principalDf = pd.DataFrame(data = principalComponents, columns = col_name[0:dim])

principalDf.head(5)


df[['target']].head()

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
finalDf.head(5)

# Explained Variance
pca.explained_variance_ratio_

with open('PCA_feature_5_dim_3_temp.pkl', 'wb') as f:
    pickle.dump(finalDf, f)


# Step 4: Visualize 2D Projection

if dim == 2:
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 Component PCA', fontsize = 20)


    targets = ['easy', 'difficult']
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
else:
    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection ="3d")
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('3 Component PCA', fontsize = 20)


    targets = ['easy', 'difficult']
    colors = ['r', 'g']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter3D(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , finalDf.loc[indicesToKeep, 'principal component 3']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()


d = 1