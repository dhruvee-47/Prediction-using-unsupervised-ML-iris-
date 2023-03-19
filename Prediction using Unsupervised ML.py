#!/usr/bin/env python
# coding: utf-8

# In[3]:


#loading all the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[41]:


#install.packages("ggplot2") for basic data visualization 
plt.style.use('ggplot')


# In[42]:


#Reading Iris dataset csv file 
data = pd.read_csv("C:/Users/Dhruvee Vadhvana/Documents/Internship/Iris.csv")
data 


# In[43]:


data.head()


# In[44]:


data.tail()


# In[45]:


from pandas.plotting import andrews_curves
andrews_curves(data.drop("Id", axis=1), "Species")
plt.show()


# In[12]:


#Dropping ID column 
data.drop(columns = "Id" , inplace = True)
data.head(5)


# In[13]:


#Checking the number of missing values in the dataset.
data.isnull().sum()


# In[14]:


#Finding basic information about the DataFrame
data.info()


# In[46]:


#Visualizing the data 
data.plot(kind="scatter", x="SepalLengthCm",   y="SepalWidthCm")
plt.show()


# In[16]:


x_data = data
x = data.iloc[:,0:4].values
x


# In[17]:


#Line Plot 
x_data.plot(kind = "line")


# In[22]:


#The simplest invocation uses scatterplot() for each pairing of the variables
#histplot() for the marginal plots along the diagonal:
sns.pairplot(data, hue="Species")


# In[24]:


#clustering data points using Kmeans
from sklearn.cluster import KMeans


# In[26]:


# WCSS is the sum of squares of the distances of each data point in all clusters to their respective centroids
wcss=[] 


# In[27]:


for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=0)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
wcss


# In[29]:


#Elbow Method- to find value of k
plt.plot(range(1,11),wcss , marker='o' ,  markerfacecolor='black')
plt.title('IRIS')
plt.xlabel('no of clusters')
labels = ["Number Of Clusters" , "Wcss"]
plt.ylabel('wcss') 
# Within cluster sum of squares   
# wcss is low for higher no. of clusters
plt.legend(labels=labels)
plt.show()

We can clearly see why it is called 'The elbow method' from the above graph, the optimum clusters is where the elbow occurs. This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration. 
From this we choose the number of clusters as 3.
# In[30]:


#Clustering
kmeans=KMeans(n_clusters=3,init="k-means++",max_iter=300,n_init=10,random_state=0)
identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[31]:


#To see the cluster centroids as a list we use : 
kmeans.cluster_centers_


# In[36]:


# Visualising the clusters - On the first two columns
plt.scatter(x[identified_clusters  == 0, 0], x[identified_clusters  == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[identified_clusters  == 1, 0], x[identified_clusters  == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[identified_clusters  == 2, 0], x[identified_clusters  == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()

