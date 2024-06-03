import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Unblock?
#from kneed import KneeLocator
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import ClusterHelper as ch
import importlib
import plotly.express as px
import plotly.graph_objects as go
importlib.reload(ch)
import streamlit as st
import mpld3
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

#Data Preparation

df = pd.read_csv(r'acs_demographic_data_by_census_tract.csv')
output_df = ch.prepareDataOneProduct(df)

#What are we looking at


#Data Cleaning


#Effects of Outliers

kmeans_set = output_df.copy()
kmeans_set = ch.outlierAddition(kmeans_set)
kmeans_set = kmeans_set.loc[(kmeans_set!=0).any(axis=1)]

#Init sklearn objects
Sc = StandardScaler()
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init='auto',
    max_iter=3000
)
#Run pca to find plottable data
pca = PCA(n_components=3)

#Fit pandas dataframe
X = Sc.fit_transform(kmeans_set)

#Fit PCA and kmeans
kmeans.fit(X)
pca_data = pd.DataFrame(pca.fit_transform(X), columns=['PC1','PC2', 'PC3'])
pca_data['cluster'] = pd.Categorical(kmeans.labels_)

#Clusters plotted against PCA axises
#%matplotlib tk
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure= False)
fig.add_axes(ax)

sc = ax.scatter(pca_data['PC1'], pca_data['PC2'], pca_data['PC3'], s = 40, alpha = 1, c= pca_data['cluster'])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Kmeans With Outliers')
# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

#UNDO THIS COMMENT
#plt.show()




#What happens when outliers are removed from dataset


kmeans_set_without_outliers = output_df.copy()



#Init sklearn objects
Sc = StandardScaler()
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init='auto',
    max_iter=3000
)
#Run pca to find plottable data
pca = PCA(n_components=3)

#Fit pandas dataframe
X = Sc.fit_transform(kmeans_set_without_outliers)


#Fit PCA and kmeans
kmeans.fit(X)
pca_data = pd.DataFrame(pca.fit_transform(X), columns=['PC1','PC2', 'PC3'])
pca_data['cluster'] = pd.Categorical(kmeans.labels_)
pca_data_plot = pca_data.copy()
pca_data_plot.loc[pca_data['cluster'] == 0,'cluster'], pca_data_plot.loc[pca_data['cluster'] == 1,'cluster'], pca_data_plot.loc[pca_data['cluster'] == 2,'cluster'] = 2, 1, 0 

#%matplotlib tk
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure= False)
fig.add_axes(ax)

sc = ax.scatter(-1*pca_data_plot['PC1'], pca_data_plot['PC2'], pca_data_plot['PC3'], s = 40, alpha = 1, c= pca_data_plot['cluster'])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')


# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

#UNDO THIS COMMENT
#plt.show()

#What does a cluster consist of


#print(kmeans_set_without_outliers.head())

#DBSCAM

dbscan = DBSCAN(eps = 5.25, min_samples= 2).fit(X)
pca_data['cluster'] = pd.Categorical(dbscan.labels_)

#%matplotlib tk
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure= False)
fig.add_axes(ax)

sc = ax.scatter(pca_data['PC1'], pca_data['PC2'], pca_data['PC3'], s = 40, alpha = 1, c= pca_data['cluster'])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

#UNDO THIS COMMENT
#plt.show()


#OPTICS


optics = OPTICS(min_samples=2).fit(X)
pca_data['cluster'] = pd.Categorical(dbscan.labels_)

#%matplotlib tk
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure= False)
fig.add_axes(ax)

sc = ax.scatter(pca_data['PC1'], pca_data['PC2'], pca_data['PC3'], s = 40, alpha = 1, c= pca_data['cluster'])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

# legend
plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)

#UNDO THIS COMMENT
#plt.show()

#Temporal dependance in Sales


#Multiple products at once

#Apply time series data
df = pd.read_csv(r'acs_demographic_data_by_census_tract.csv')
groupedProductData = ch.createBulkData(df)

#What does a cluster consist of
#print(kmeans_set_without_outliers.head())



importlib.reload(ch)
#UNDO THIS COMMENT
#ch.clusterSpecificProduct(groupedProductData, 0)


#UNDO THIS COMMENT
#ch.clusterSpecificProduct(groupedProductData, 1)

# Considerations for parallelizasion of clusters

st.title("Clustering Demo")
st.subheader("Customer segmentation in Oslo")

#Adding a html file
HtmlFile = open("test.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
#print(source_code)
components.html(source_code, width=500, height = 500)
'''
Red cluster:
* High income, frugal spending, wide age range
* “Affluent neighborhood”

Yellow cluster:
* Low income, low spending, old age
* “Minstepensjonist/uføretrygdet”

Purple:
* Low income, high spending, young
* “Students”

'''



#st.plotly_chart(ch.clusterSpecificProduct(groupedProductData, 0), use_container_width = False, height = 300, width = 150)
st.plotly_chart(ch.clusterSpecificProduct(groupedProductData, 0), use_container_width = False)
'''
- Cluster 0: Low income area, small/medium stores, consisting mostly of students aged 20-30, Politics: 30% AP, 10% V, 25 MDG, 15% H.
- Cluster 1: Medium income area, large stores, consisting of families with children, 86% employment rate, Politics: 30% H, 20% AP, 5% V.
- Cluster 2: Older couples aged 60 and above, medium/large stores, 38% employment rate, median income 450k NOK.
'''

st.plotly_chart(ch.clusterSpecificProduct(groupedProductData, 1), use_container_width = False)
'''
- Cluster 0: Low income area, consisting mostly of students aged 20-30, Politics: 30% AP, 10% V, 25 MDG, 15% H.
- Cluster 1: Low income area, low emplyment rate 23%, Politics: 30% FRP, 20% AP, 5% V.
- Cluster 2: Affluent area, median income above 1.2M NOK.
- Cluster 3: Medium income area, consisting of families with children, 86% employment rate, Politics: 30% H, 20% AP, 5% V.
- Cluster 4: Older couples aged 60 and above, 38% employment rate, median income 450k NOK.
'''



