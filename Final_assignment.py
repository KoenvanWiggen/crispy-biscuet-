#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 09:28:04 2024

@author: k.g.vanwiggen
"""

############################## Import Packages ################################
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import rasterio
from rasterio.merge import merge
from rasterio.windows import Window
import sklearn
from rasterio.enums import Resampling
from pyproj import Transformer
from shapely.geometry import Polygon
from shapely.geometry import box
from shapely.geometry import mapping


############################## Adjust the image ###############################
# Normalize values based on the cut

def normalize_cut(data, cloud_threshold, flood_threshold):
    cut_min = np.min(data) 
    cut_max = np.max(data) 
    normalized_data = ((data - cut_min) / (cut_max - cut_min)) * 250
    # Removing clouds
    normalized_data = np.where(data > cloud_threshold, 100, data)
    # Let the flooded areas pop out
    normalized_data = np.where(normalized_data > flood_threshold, 250, normalized_data)
    return normalized_data

def plot_raster(data, title):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.imshow(data)
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.show()

def resample(text1, method, text2, factor, adjust_window, start_col, start_row, width, height, adjust_cut, cloud_threshold, flood_threshold):
    with rasterio.open(text1) as raster_src:
        # Define the window based on input
        if adjust_window:
            window = Window(
                col_off=start_col,
                row_off=start_row,
                width=width,
                height=height
            )
        else:
            window = Window(
                col_off=0,
                row_off=0,
                width=raster_src.width,
                height=raster_src.height
            )
        
        # Read the data for the specified window
        original_data_red = raster_src.read(1, window=window)
        original_data_green = raster_src.read(2, window=window)
        original_data_blue = raster_src.read(3, window=window)

        
        # Calculate the new dimensions for resampling
        new_width = window.width // factor
        new_height = window.height // factor
        
        # Resample the data
        resampled_data_red = raster_src.read(1, out_shape=(new_height, new_width),
            resampling=method, window=window)
        resampled_data_green = raster_src.read(2, out_shape=(new_height, new_width),
            resampling=method, window=window)
        resampled_data_blue = raster_src.read(3, out_shape=(new_height, new_width),
            resampling=method, window=window)
        
        original_data = np.dstack((original_data_red, original_data_green
                                   , original_data_blue))
        resampled_data = np.dstack((resampled_data_red, resampled_data_green, 
                                 resampled_data_blue))
        
        if adjust_cut: 
            original_data = normalize_cut(original_data, cloud_threshold, flood_threshold) 
            resampled_data = normalize_cut(resampled_data, cloud_threshold, flood_threshold)
        
        # Adjust the transform for the window
        window_transform = raster_src.window_transform(window)
        scaled_transform = window_transform * window_transform.scale(
            (window.width / new_width),
            (window.height / new_height))
        
        # Update the profile for the output file
        resampled_crs = raster_src.profile.copy()
        resampled_crs.update({"width": new_width,"height": new_height,
            "transform": scaled_transform,})
    
    resampled_data = np.moveaxis(resampled_data, -1, 0)  # Change to (bands, height, width)
    # Create the output file with the updated profile
    with rasterio.open(text2, "w", **resampled_crs) as dst: 
        dst.write(resampled_data)
    
    return original_data, resampled_data
    
# The first piece
Greece0_before, _= resample('Greece_image.jp2', Resampling.average, 'Greece_adjusted.jp2', 1, False, 0, 0, 0, 0, False, 250, 180)
plot_raster(Greece0_before, 'Piece one')
#plot_raster(Greece0_adjusted, 'Greece adjusted one')

# The second piece
Greece1_before, _= resample('Greece_image1.jp2', Resampling.average, 'Greece_adjusted1.jp2', 1, False, 0, 0, 0, 0, False, 250, 130)
#plot_raster(Greece1_before, 'Piece two')
#plot_raster(Greece1_adjusted, 'Greece adjusted two')

# The third piece
Greece2_before, _= resample('Greece_image2.jp2', Resampling.average, 'Greece_adjusted2.jp2', 1, False, 0, 0, 0, 0, False, 250, 200)
#plot_raster(Greece2_before, 'Piece three')
#plot_raster(Greece2_adjusted, 'Greece adjusted three')

# The fourth piece
Greece3_before, _= resample('Greece_image3.jp2', Resampling.average, 'Greece_adjusted3.jp2', 1, False, 0, 0, 0, 0, False, 250, 200)
#plot_raster(Greece3_before, 'Piece four')
#plot_raster(Greece3_adjusted, 'Greece adjusted four')

# The fifth piece
Greece4_before, _= resample('Greece_image4.jp2', Resampling.average, 'Greece_adjusted4.jp2', 1, False, 0, 0, 0, 0, False, 250, 200)

# The sixt piece
Greece5_before, _= resample('Greece_image5.jp2', Resampling.average, 'Greece_adjusted5.jp2', 1, False, 0, 0, 0, 0, False, 250, 200)

def mosaic_rasters(output_path, *input_files):
    # List to store open rasters
    src_files = []
    
    for file in input_files:
        src = rasterio.open(file)
        src_files.append(src)
    
    # Merge rasters (default takes first non-null value)
    mosaic_data, mosaic_transform = merge(src_files)
    
    # Update the metadata for the output file
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "JP2OpenJPEG",
        "height": mosaic_data.shape[1],
        "width": mosaic_data.shape[2],
        "transform": mosaic_transform,
    })
    
    # Write the mosaic to the output file
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic_data)
    
    # Close all source files
    for src in src_files:
        src.close()
    
    return mosaic_data

#Perform the mosaic
# QGIS image
together = mosaic_rasters(
    "Greece_mosaic.jp2", 'Greece_image.jp2', 'Greece_image1.jp2', 
    'Greece_image2.jp2', 'Greece_image3.jp2', 'Greece_image4.jp2', 
    'Greece_image5.jp2',)

plot_raster(together, 'All the pieces together')

final_before, _= resample("Greece_mosaic.jp2", Resampling.average, 'final_before.jp2', 1, True, 3000, 3500, 6000, 4000, False, 300, 200)
_, final_adjusted= resample("Greece_mosaic.jp2", Resampling.average, 'final_adjusted.jp2', 40, True, 3000, 3500, 6000, 4000, False, 300, 200)
plot_raster(final_before, 'Final data before resampling')
plot_raster(final_adjusted, 'Final data after resampling')

def flood_difference(data, flood_threshold):
    flooded_data = np.where(data >= flood_threshold, 250, data)
    flooded_data = np.where(flooded_data < flood_threshold, 0, flooded_data)
    return flooded_data

#flooded = flood_difference(final_adjusted, 200)
#plot_raster(flooded, 'Final data before resampling')










