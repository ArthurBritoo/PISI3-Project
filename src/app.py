from src.models.data_loader import CSVDataLoader
from src.controllers.clustering_controller import ClusteringController
from src.views.clustering_view import ClusteringVisualization
from src.utils.data_cleaning import clean_price_data, validate_coordinates

def main():
    # Initialize components
    data_loader = CSVDataLoader('quintoandar_recife.csv')
    visualization = ClusteringVisualization()
    controller = ClusteringController(data_loader, visualization)
    
    # Perform clustering analysis
    data = controller.perform_clustering(n_clusters=5)
    
    # Create visualizations
    map_plot, price_plot = controller.visualize_clusters(data)
    
    # Show plots
    map_plot.show()
    price_plot.show()

if __name__ == '__main__':
    main()