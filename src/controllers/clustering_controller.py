from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class ClusteringController:
    def __init__(self, data_loader, visualization):
        self.data_loader = data_loader
        self.visualization = visualization
        self.model = None
        self.scaler = StandardScaler()
        
    def preprocess_data(self, df):
        """
        Preprocess the data for clustering
        """
        features = ['price', 'area', 'latitude', 'longitude']
        return self.scaler.fit_transform(df[features])
        
    def perform_clustering(self, n_clusters=5):
        """
        Perform K-means clustering on the data
        """
        data = self.data_loader.load_data()
        X = self.preprocess_data(data)
        
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.model.fit_predict(X)
        
        # Add clusters to the dataframe
        data['cluster'] = clusters
        return data
    
    def visualize_clusters(self, data):
        """
        Create visualizations for the clustering results
        """
        map_plot = self.visualization.plot_clusters(data)
        price_plot = self.visualization.plot_price_distribution(data)
        return map_plot, price_plot