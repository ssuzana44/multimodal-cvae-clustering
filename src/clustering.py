#Clustering done along with data in dataset.py 
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering

def run_kmeans(data, k=2):
    return KMeans(n_clusters=k, random_state=42).fit_predict(data)

def run_dbscan(data, eps=3.0, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)

def run_agglomerative(data, k=2):
    return AgglomerativeClustering(n_clusters=k).fit_predict(data)

def run_spectral(data, k=2):
    return SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42).fit_predict(data)