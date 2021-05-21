#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Labeling dataset into X and Y
dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Python/Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values

# Replacing missing values with colume average 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan,strategy= 'mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Convert categorical dataset into binary digits
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers= [('Encoder', OneHotEncoder(),[0])], remainder= 'passthrough')
X = np.array(ct.fit_transform(X))

# Convert y dataset into 0's and 1's
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

# Split dataset into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = 0.20, random_state= 1)

# Standardized x training and testing dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:,3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])


# In[5]:


#Simple linear regression 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Labeling dataset into X and Y
dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression//Section 4 - Simple Linear Regression/Python/Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Split dataset into training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

# Creating a Linear regression line using training dataset
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

# Displaying regression line w/test dataset 
plt.scatter(x_test, y_test, color = 'green')
plt.plot(x_train, reg.predict(x_train), color = 'red')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Making single prediction
# print(regressor.predict([[12]]))

# To get coef and y-int of the regression line, use the following lib
# print(regressor.coef_)
# print(regressor.intercept_)


# In[29]:


# Multiple Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('Encoder', OneHotEncoder(), [3])], remainder= 'passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.20, random_state= 0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
np.set_printoptions(precision=2)
np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)

# Making single prediction 
# R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000 and State = California
# 1 0 0 is for California 
# reg.predict([[1, 0, 0, 160000, 130000, 300000]])

# To get coef and y-int of the regression line, use the following lib
# print(regressor.coef_)
# print(regressor.intercept_)


# In[24]:


#Polynomial linear model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Labeling dataset into X and Y
dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# You don't need to build a linear model for polynomial
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree= 5) #Changing the degree will train the data all the way up to the degree
x_poly = poly_reg.fit_transform(x) #Creates the x^n column
reg_2 = LinearRegression()
reg_2.fit(x_poly,y)

plt.scatter(x,y,color= 'red')
plt.plot(x, reg_2.predict(poly_reg.fit_transform(x)), color= 'green')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# To predict a single value
# reg_2.predict(poly_reg.fit_transform([[6.5]]))


# In[13]:


# Support Vector Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Labeling dataset into X and Y
dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Python/Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)

# Feature scaling b/c dependant varible range(45K-100K) is >> then independant variable(1-10)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler() #Use different StandardScaler for x and y because it will compute mean and SD, which are different for both
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel= 'rbf')
regressor.fit(x,y)

# To predict single value
regressor.predict(sc_x.transform([[6.5]])) #This will predict the value but in scaling
# To convert back to original format, use
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))

plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y),color= 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color= 'green')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# For higher resolution and smoother curve
# x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(X)), 0.1)
# x_grid = x_grid.reshape((len(x_grid), 1))
# plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
# plt.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid))), color = 'blue')
# plt.title('Truth or Bluff (SVR)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()


# In[23]:


# Decision Tree regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 8 - Decision Tree Regression/Python/Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state= 1)
regressor.fit(x,y)

regressor.predict([[6.5]])

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (DTR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[32]:


# Random Forest Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 9 - Random Forest Regression/Python/Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators= 10, random_state=0)
regressor.fit(x,y)

regressor.predict([[6.5]])

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(x,y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Tree)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# In[29]:


# Logistic Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv')
x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state= 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 1)
classifier.fit(x_train,y_train)

# Predicting single point 
classifier.predict(sc.transform([[30,87000]]))

# Test set results
y_pred = classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)


# Logistic classifer is a linear classifer; therefore, you will have a straight line.


# In[33]:


# KNN classifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv')
x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state= 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc = accuracy_score(y_test, y_pred)
print(acc)

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 1),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 1))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[34]:


# SVC 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv')
x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, random_state= 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel= 'linear', random_state= 0) 
# For non-linearly separated data, change the kernel function i.e.'rbf', 'sigmoid'
classifier.fit(x_train, y_train)


y_pred = classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
acc = accuracy_score(y_test,y_pred)
print(acc)

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('yellow', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[11]:


# Naive Bayes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv')
x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state= 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)

classifier.predict(sc.transform([[30,87000]]))

y_pred = classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
acc = accuracy_score(y_test,y_pred)
print(acc)

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('yellow', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[16]:


# Decision Tree Classification 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv')
x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state= 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(criterion= 'entropy', random_state=0)
classifier.fit(x_train, y_train)

classifier.predict(sc.transform([[30,87000]]))

y_pred= classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)

from sklearn.metrics import confusion_matrix, accuracy_score
cm= confusion_matrix(y_test, y_pred)
print(cm)
acc = accuracy_score(y_test, y_pred)
print(acc)

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('yellow', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[35]:


# Random Forest Classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv')
x = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.20, random_state= 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators= 100, criterion= 'entropy', random_state= 0)
classifier.fit(x_train, y_train)

classifier.predict(sc.transform([[30,87000]]))

y_pred= classifier.predict(x_test)
np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)

from sklearn.metrics import confusion_matrix, accuracy_score
cm= confusion_matrix(y_test, y_pred)
print(cm)
acc = accuracy_score(y_test, y_pred)
print(acc)

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(x_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('yellow', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[2]:


# Classification self-practice 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Classification/Data.csv')
x = dataset.iloc[:,:-1]
y = dataset.iloc[:, -1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.20, random_state= 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)

# y_pred= classifier.predict(x_test)
# np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


# K-Mean Clustering 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 4 - Clustering/Section 24 - K-Means Clustering/Python/Mall_Customers.csv')
x = dataset.iloc[:,3:].values

# To evaluate k number of clusters to choose
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state= 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_) # inertia is an kmean attribute used to compute wcss values
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

from sklearn.cluster import KMeans
kmeans= KMeans( n_clusters= 5, init= 'k-means++', random_state= 1)
y_kmeans = kmeans.fit_predict(x) #trains as well as returns the value for which cluster the value belongs to 

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0,1], s = 50, c = 'red', label = 'Cluster 1') #x[cust belonging to clu 0, column I wanted (Income)
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 50, c = 'magenta', label = 'Cluster 5')
# An attribute used to locate the centroid. 2D array where [different centroid, coordinate] 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 4 - Clustering/Section 24 - K-Means Clustering/Python/Mall_Customers.csv')
x = dataset.iloc[:,[3,4]]

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward')) #ward is used to minimize the function
plt.title('Dendrogram')
plt.xlabel('Customers') # observational points
plt.show()


# In[21]:


# Association Rule Learning 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Make sure to indicate that there are no headers
dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 5 - Association Rule Learning/Section 28 - Apriori/Python/Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

from apyori import apriori
rules = apriori(transactions= transactions, min_support= 0.003, min_confidence= 0.2, min_lift= 3, min_length =2, max_length = 2)

#Output stright from apriori function
results = list(rules)

#Pandas lib output
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
# This will sort lift in desc order
resultsinDataFrame.nlargest(n = 10, columns = 'Lift')

# Eclat model 
# Similar to apriori model but you only have support.... You are looking about combinations(watched both movies) rather than pattern of movies watched


# In[ ]:





# In[6]:


# Upper Confidence Bound
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math 

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Python/Ads_CTR_Optimisation.csv')

N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 10000000
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.show()


# In[16]:


# Thompson Sampling
# Compare to UCB (Deterministic), Thompson Sampling behaves in probabilistic manner
# Can account for delay feedback, say like 500 input, whereas UCB needs to be updated every round
# Better empirical evidence 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Python/Ads_CTR_Optimisation.csv')

# Note the round used and efficiency of this method 
N = 2000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()


# In[3]:


## Natural Language Processing 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import re
## lib used to ignore words that does not give a hint whether the review is + or -
import nltk
nltk.download('stopwords')

#Import words to our code
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer ##  consist of taking root of words that will tell us about the result.

## Quoting = 3 will tell the model to ignore "" in your text
dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Python/Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if word not in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1500) ## This will consider 1500 frequent words from total of 1566 words 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[45]:


## Artificial Neural Network
import numpy as np
import pandas as pd
import tensorflow as tf

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Python/Churn_Modelling.csv')
x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers= [('Encoder', OneHotEncoder(),[1])], remainder= 'passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.20, random_state= 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Building the ANN
ann = tf.keras.models.Sequential()
# Hidden layers, use rectifier funtion
ann.add(tf.keras.layers.Dense(units= 6, activation= 'relu')) ## You always need a function (use rectifier or relu ) when implementing a layer
ann.add(tf.keras.layers.Dense(units= 6, activation= 'relu'))

# Output, use sigmoid function (probability)
ann.add(tf.keras.layers.Dense(units= 1, activation= 'sigmoid')) ## For units/neurons, use number of bits. For example, you need 3 units for a(1 0 0), b(0 1 0), c(0 0 1)

## Compiling ANN
## Gradiant descent(update weight after batch) vs stochastic gradiant descent(update after each row)
## Stochastic performs better than gradiant descent (optimizer = adam)
ann.compile(optimizer= 'adam', loss= 'binary_crossentropy' , metrics= ['accuracy'] ) ##when doing binary output, loss = binary_crossentropy ... For non-binary, loss = categorical_crossentropy..... NOte: For regression, the loss is root mean square

## Training the data
ann.fit(x_train, y_train, batch_size= 32, epochs= 100) ## batch_size is compare expected with real in a batch.. epoch is the number of iteration for ANN to train

# ann.predict(sc.transform([[1,0,0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5
## the output will be a probability due to sigmoid function.. To get True or F, use > 0.5

y_pred = ann.predict(x_test)
#Binary output, so you also need binary y_pred
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[43]:


ann.predict(sc.transform([[1,0,0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))  


# In[ ]:





# In[ ]:





# In[16]:


#Convolutional Neural Networks (CNN) -- ability to recognize features from an image(probability) 

# Steps: Convolution --> Max Pooling --> flattening --> Full Connection
#

# Convolution Layer --> series of feature map (filter) image, which is obtained from the original image 
    #could you nose in one feature detector(3x3), or ears in other one. 
        #The purpose of feature detector is to obtain maximum pixel numbers from an image.
        #Maximum feature detector means the main focus is on that part of the image

# Rectifier function is used to increase non-linearity because images themselves have non-linearity such as background and objects together 
    # Transition between pixels (feature detector). Negative pixels means background and positive means popping out 

# Max pooling(2x2) --> taking the maxing value from a feature map
 #  Accounts for distortion while preserving the features (numbers) and elimination large percent of not relevent info

# Flattening --> Take numbers from pooled map (cube) and flatten each element,like a string, row by row.

# Full connection --> This is where you will build ANN 
# May have more than 1 output since it is a classification of features

#Softmax function --> adds the probqbility of all your outputs to 1
#Cross-entropy --> helps change weight using log function to make adjectment

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator 

#Preprocessing training set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # Importing images to the notebook
training_set = train_datagen.flow_from_directory(
        'Desktop/Section 40 - Convolutional Neural Networks (CNN)/dataset/training_set/',
        target_size= (64, 64),
        batch_size= 32,
        class_mode='binary') ## binary or categorical 

#Preprocessing test set
test_datagen = ImageDataGenerator(rescale=1./255)
    # Importing images to the notebook
test_set = test_datagen.flow_from_directory(
        'Desktop/Section 40 - Convolutional Neural Networks (CNN)/dataset/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#Building the CNN
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= 3, activation= 'relu', input_shape= (64,64,3)))
cnn.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides= 2))## pool_size is that dim of your pixel you extract from the original image and stride is how many pixel you will be moving

## Adding another layer
cnn.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= 3, activation= 'relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size= 2, strides= 2))

# Flattening layer
new_input = cnn.add(tf.keras.layers.Flatten())

#full connection
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units= 128, activation= 'relu'))
ann.add(tf.keras.layers.Dense(units= 128, activation= 'relu'))

#output layer
ann.add(tf.keras.layers.Dense(units= 1, activation= 'sigmoid'))

#Training the dataset
cnn.compile(optimizer='adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
# cnn.fit(x = training_set, validation_data= test_set, epochs= 25)

import numpy as np
from keras.preprocessing import image

## Making single prediction
    # Importing test image
test_image = image.load_img('single_prediction/cat_or_dog_1.jpg', target_size= (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis= 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Dog'
else:
    prediction = 'Cat'


# In[18]:


# Dimensionality Reduction: 2 types; Feature selection and Feature Extraction
# whatever is the original number of our independent variables, we can often end up with two independent variables by applying an appropriate Dimensionality Reduction technique.

# In this tab, we will be covering Feature Extraction, which consist of 4 parts
# 1) Principle Component Analysis 
#    Identity pattern and detect correlation b/w variables
#    Reduce d-dim by project onto k-subspace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)/Python/Wine.csv')
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components= 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0)
classifier.fit(x_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
ac = accuracy_score(y_test, y_pred)
print(ac)

from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('yellow', 'purple', 'orange'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# In[26]:


# 2) Linear Discriminant Analysis
# Similar to PCA but we are interested in axes that maximize the separation b/w multiple classes
# They will project d-dim into separated k-dims

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('cSection 43 - Principal Component Analysis (PCA)/Python/Wine.csv')
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components= 2)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state= 0)
classifier.fit(x_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
ac = accuracy_score(y_test, y_pred)
print(ac)

from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('yellow', 'blue', 'orange'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()


# In[29]:


# Model Selection
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 10 - Model Selection _ Boosting/Section 48 - Model Selection/Python/Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

# K-fold cross validation
from sklearn.model_selection import cross_val_score
## Estimator is the type of mode you're using 
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10) #cv -> split train data into 10 train fold and it will train and test each of the fold
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

#Grid search -> Finds the best accuracy from the parameter given. You can change the parameters to find more accurate model
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[5]:


## XGBoost ->
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('/Users/mithil29/Desktop/Machine Learning A-Z (Codes and Datasets)/Part 10 - Model Selection _ Boosting/Section 49 - XGBoost/Python/Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 1)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(x_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

from sklearn.model_selection import cross_val_score
cvs = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(cvs.mean()*100))
print("Standard Deviation: {:.2f} %".format(cvs.std()*100))


# In[ ]:




