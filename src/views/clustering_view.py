import plotly.express as px
import plotly.graph_objects as go

class ClusteringVisualization:
    @staticmethod
    def plot_clusters(df, cluster_column='cluster', lat_column='latitude', lon_column='longitude'):
        """
        Create an interactive scatter plot of property clusters on a map
        """
        fig = px.scatter_mapbox(
            df,
            lat=lat_column,
            lon=lon_column,
            color=cluster_column,
            zoom=11,
            title='Property Clusters in Recife'
        )
        fig.update_layout(mapbox_style='carto-positron')
        return fig

    @staticmethod
    def plot_price_distribution(df, cluster_column='cluster', price_column='price'):
        """
        Create a box plot of price distribution by cluster
        """
        fig = px.box(
            df,
            x=cluster_column,
            y=price_column,
            title='Price Distribution by Cluster'
        )
        return fig