import pandas as pd
import numpy as np

def clean_price_data(df, price_column='price'):
    """
    Clean and validate price data
    """
    df = df.copy()
    # Remove invalid prices (negative or zero)
    df = df[df[price_column] > 0]
    # Remove outliers using IQR method
    Q1 = df[price_column].quantile(0.25)
    Q3 = df[price_column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[price_column] < (Q1 - 1.5 * IQR)) | (df[price_column] > (Q3 + 1.5 * IQR)))]
    return df

def validate_coordinates(df, lat_column='latitude', lon_column='longitude'):
    """
    Validate geographical coordinates
    """
    # Recife approximate coordinates boundaries
    lat_min, lat_max = -8.2, -7.9
    lon_min, lon_max = -35.0, -34.8
    
    return df[
        (df[lat_column] >= lat_min) & (df[lat_column] <= lat_max) &
        (df[lon_column] >= lon_min) & (df[lon_column] <= lon_max)
    ]