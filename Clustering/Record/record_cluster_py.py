import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hclust
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import silhouette_score


##### -------------------- add label manually -------------------- #####
df = pd.read_csv('/Users/lingfengcao/Leyao/ANLY501/HW MOD03/forestfires.csv')
# print(df.describe())

## Copy the original dataframe for labeling
## drop columns with text data
df.drop(['month','day'],axis=1, inplace=True)
new_df = df

## look at the distribution of the area column
## and determine labeling standards
# new_df.hist(column='area',bins=100)
# plt.show()

## add label and create a new column for labeling
def label_area (row):
    if row['area'] <= 6.57 :
        return 'small'
    if row['area'] <= 15 :
        return 'meduim'
    else:
        return 'large'

new_df['area_label']=new_df.apply (lambda row: label_area(row), axis=1)
trueLabel = new_df['area_label']


## save the new_df
new_df.to_csv('/Users/lingfengcao/Leyao/ANLY501/HW MOD03/labeled_forestfires.csv')


##### -------------------- normalization and heatmap -------------------- #####
df.drop(['area_label'],axis=1,inplace=True)

# normalize the data
# and visualize with heatmap

scaler = preprocessing.MinMaxScaler()
df_normal = scaler.fit_transform(df)
print(pd.DataFrame(df_normal).describe())
sns.clustermap(df_normal[:,2:10],yticklabels=trueLabel)
plt.show()


##### -------------------- Determine number of clusters -------------------- #####

##### -------------------- Using Elbow Method -------------------- #####
## run kmeans for k values from 1 to 10
## to determine the best number of clusters
## using elbow method
model_slct = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df)
    model_slct.append(kmeans.inertia_)
plt.plot(range(1, 10), model_slct)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('model_slct')
plt.show()

## based on the elbow method
## I choose the number of clusters to be 2 or 3

##### -------------------- Using Silhouette Method -------------------- #####
from yellowbrick.cluster import SilhouetteVisualizer

fig, ax = plt.subplots(2, 2, figsize=(15,8))
for i in [2, 3, 4, 5]:
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(df)
visualizer.show()
## based on the average Silhouette for k = 2, 3, 4, 5
## k = 2 gives highest silhouette score
## however I will still choose k=3 for study purposes

##### -------------------- Compare distance metrics -------------------- #####

km_n2 = KMeans(n_clusters=2)
km_n2.fit_predict(df)
# euclidean distance
score_euc = silhouette_score(df, km_n2.labels_, metric='euclidean')
# cosine distance
score_cos = silhouette_score(df, km_n2.labels_, metric='cosine')
# manhattan distance
score_manh = silhouette_score(df, km_n2.labels_, metric='manhattan')
print(score_euc, score_cos, score_manh)

## based on the results, cosine similarity metric gives highest silhouette score
## of 0.783
## while the euclidean distance metric has a score of 0.726
## which is pretty close

##### -------------------- perform k (k=3) means cluster analysis -------------------- #####
kmeans = KMeans(n_clusters=3).fit(df_normal)
labels = pd.DataFrame(kmeans.labels_)
labeledColleges = pd.concat((df,labels),axis=1)
labeledColleges = labeledColleges.rename({0:'labels'},axis=1)
labeledColleges.to_csv('/Users/lingfengcao/Leyao/ANLY501/HW MOD03/km_labeled_forestfires.csv')

sns.lmplot(x='area',y='DC',data=labeledColleges,hue='labels',fit_reg=False)
plt.title("K-Means Clustering with k=3 ")


##### -------------------- hierarchical clustering -------------------- #####
dendrogram = hclust.dendrogram(hclust.linkage(df, method  = "ward"))
plt.title('Dendrogram')
plt.xlabel('forestfires')
plt.ylabel('Euclidean distances')
plt.show()

## based on the hierarchical clustering result
## the best number of clusters would be 2
