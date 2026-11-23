import os
import re
import inspect
import sys
from collections import defaultdict
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pyinterp
import pyinterp.fill as fill
import xarray as xr

import s3fs 
import pandas as pd
import cv2
from natsort import natsorted
import glob
import logging
from shapely import geometry
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

# =============================================================================
# extract_xarray_in_region
# =============================================================================


def extract_xarray_in_region(directory, area):
    """
    Extracts xarray datasets from SWOT NetCDF data for a specific region
    

    Parameters
    ------------
    directory : str
        Path to the directory containing the NetCDF files
    area : list
        List with the boundaries of the region of interest [longitude_min, latitude_min, longitude_max, latitude_max]
    

    Returns
    ---------
    datasets : Dict
        Dictionary containing the xarray.Datasets for the region
   
    """

    lon_min, lat_min, lon_max, lat_max = area
    datasets = {}
    i = 0

    variables_to_load = ["ssha", "mdt", "latitude", "longitude","quality_flag","time"]
    files_in_dir = os.listdir(directory)

    for filename in files_in_dir:
        file_path = os.path.join(directory, filename)

        ds_tmp = xr.open_dataset(file_path, chunks={})
        variables_to_drop = [var for var in ds_tmp.variables if var not in variables_to_load]
        ds_tmp.close()
        del ds_tmp

        # Open the file (lazy loading) excluding unnecessary variables
        ds = xr.open_dataset(file_path, chunks={}, drop_variables=variables_to_drop)

        if ds:
            if lon_min < lon_max:
                selection = (

                        (ds['latitude'] >= lat_min) &
                        (ds['latitude'] <= lat_max) &
                        (ds['longitude'] >= lon_min) &
                        (ds['longitude'] <= lon_max)
                )

            else:
                selection = (

                        (ds['latitude'] >= lat_min) &
                        (ds['latitude'] <= lat_max) &
                        (((ds['longitude'] >= lon_min) & (ds['longitude'] <= 360)) | (ds['longitude'] <= lon_max))

                )

            selection = selection.compute()

            ds_area = ds.where(selection, drop=True)
            ds.close()

            if ds_area['latitude'].size > 0:
                datasets[i] = ds_area
                i += 1

            ds_area.close()

    return datasets

# =============================================================================
# extract_xarrays_by_time_and_region
# =============================================================================


def extract_xarrays_by_time_and_region(database_path, start_date_str, end_date_str, area,variables_to_load=None):
    """
    Extracts xarray datasets from NetCDF data in folders within a specified date range and for a specific region.
    
    Parameters
    ----------
    database_path : str
        Path to the directory containing subfolders with NetCDF files.
    start_date_str : str
        Start date in 'YYYYMMDD' format.
    end_date_str : str
        End date in 'YYYYMMDD' format.
    area : list
        List with the boundaries of the region of interest [longitude_min, latitude_min, longitude_max, latitude_max].
    variables_to_load : list, optional
        List of variable names to load from each dataset. Defaults to ["ssha", "mdt", "latitude", "longitude"].   
        
    Returns
    -------
    combined_datasets_dict : Dict
        A dictionary of xarray Datasets combining the results for each matching folder and region.
    """
    
    lon_min, lat_min, lon_max, lat_max = area
    matching_folders = check_directory(database_path, start_date_str, end_date_str)
    combined_datasets_dict = defaultdict(list)
    current_key = 0
    
    # Set default variables to load if not provided
    if variables_to_load is None:
        variables_to_load = ["ssha", "mdt", "latitude", "longitude"]

    # Convert start_date_str and end_date_str to datetime objects for comparison
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')

    for folder in matching_folders:
        folder_path = os.path.join(database_path, folder)
        files_in_dir = os.listdir(folder_path)
        
        for filename in files_in_dir:
            file_path = os.path.join(folder_path, filename)
            
            # Open the dataset to check the `time_coverage_begin` attribute
            try:
                ds_tmp = xr.open_dataset(file_path, chunks={})
                time_coverage_begin = ds_tmp.attrs.get('time_coverage_begin')
                ds_tmp.close()
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                continue

            if time_coverage_begin:
                # Extract the date (YYYYMMDD) from time_coverage_begin (format: 'YYYY-MM-DDTHH:MM:SSZ')
                file_date_str = time_coverage_begin.split('T')[0].replace('-', '')
                file_date = datetime.strptime(file_date_str, '%Y%m%d')

                # Check if the file's date falls within the desired range
                if not (start_date <= file_date <= end_date):
                    continue  # Skip this file if it's not in the date range

            # If the file is within the desired date range, proceed with extraction
            try:
                ds_tmp = xr.open_dataset(file_path, chunks={})
                variables_to_drop = [var for var in ds_tmp.variables if var not in variables_to_load]
                ds_tmp.close()
                del ds_tmp

                ds = xr.open_dataset(file_path, chunks={}, drop_variables=variables_to_drop)

                if ds:
                    # Apply geographical selection for the dataset
                    if lon_min < lon_max:
                        selection = (
                            (ds['latitude'] >= lat_min) &
                            (ds['latitude'] <= lat_max) &
                            (ds['longitude'] >= lon_min) &
                            (ds['longitude'] <= lon_max)
                        )
                    else:
                        selection = (
                            (ds['latitude'] >= lat_min) &
                            (ds['latitude'] <= lat_max) &
                            (((ds['longitude'] >= lon_min) & (ds['longitude'] <= 360)) |
                             (ds['longitude'] <= lon_max))
                        )

                    selection = selection.compute()

                    # Extract data for the region
                    ds_area = ds.where(selection, drop=True)
                    ds.close()

                    # Add the dataset to the combined dictionary if it contains data
                    if ds_area['latitude'].size > 0:
                        combined_datasets_dict[current_key] = ds_area
                        current_key += 1

                    ds_area.close()
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue
    
    # Convert defaultdict back to a regular dictionary
    combined_datasets_dict = dict(combined_datasets_dict)
    
    return combined_datasets_dict


# =============================================================================
# count_observations
# =============================================================================


def count_observations(datasets, area, resolution):
    """
    Calculates the number of available observations per bin in the region of interest.
    
    
    Parameters
    ------------
    datasets : Dict
        Dictionary containing xarray.Datasets
    area : list
        List with the boundaries of the region of interest [longitude_min, latitude_min, longitude_max, latitude_max]
    resolution : float
        Grid resolution

    Returns
    ---------
    obs_count : np.ndarray
        Array containing the number of observations per pixel
    
    """
    lon_min, lat_min, lon_max, lat_max = area

    # Define the grid
    if lon_min > lon_max:  # if for example lon_min est 2W=358 et lon max 5E=5
        lon_grid = np.concatenate([np.arange(lon_min, 360, resolution),
                                   np.arange(0, lon_max, resolution)])
    else:
        lon_grid = np.arange(lon_min, lon_max, resolution)

    # for latitudes
    lat_grid = np.arange(lat_min, lat_max, resolution)

    # Create an array filled with zeros
    obs_count = np.full((len(lat_grid), len(lon_grid)), 0)

    # Iterating on xarrays in datasets dictionnary
    for key in range(len(datasets)):
        # print(f"{key}/{len(datasets)}")

        one_dtset = datasets[key].compute()

        # Iterating on rows/columns
        for j in range(one_dtset.ssha.shape[0]):
            for k in range(one_dtset.ssha.shape[1]):
                valeur = one_dtset.ssha[j, k]
                if not np.isnan(valeur):
                    lt = float(one_dtset.latitude[j, k])
                    ln = float(one_dtset.longitude[j, k])

                    # Find the corresponding indexes in the grid
                    lon_index = int((ln - lon_min) / resolution)
                    lat_index = int((lt - lat_min) / resolution)

                    if lon_index < 0:
                        lon_index = int((ln - (lon_min - 360)) / resolution)

                    # Increment the corresponding element in obs_count
                    obs_count[lat_index, lon_index] += 1

    one_dtset.close()
    del (one_dtset)

    return obs_count


# =============================================================================
# fill_nan
# =============================================================================


def _gauss_seidel_inplace_api():
    """
    Detect the pyinterp.fill.gauss_seidel API generation.

    pyinterp >= 2026 fills a plain float array IN PLACE and returns
    ``(iterations, residual)`` with ``epsilon`` keyword-only; older releases
    take a Grid2D and return ``(has_converged, filled_data)`` with ``epsilon``
    positional-or-keyword.

    Returns
    ---------
    bool
        True for the pyinterp >= 2026 in-place flavour.
    """
    sig = inspect.signature(pyinterp.fill.gauss_seidel)
    epsilon = sig.parameters.get("epsilon")
    return epsilon is not None and epsilon.kind == inspect.Parameter.KEYWORD_ONLY


def _gauss_seidel_fill(values, longitudes, latitudes,
                       epsilon=1e-4, num_threads=16):
    """
    Gauss-Seidel NaN filling of one swath segment, on either pyinterp generation.

    The x axis is built from the first-row longitudes taken modulo 360 and
    sorted — strictly increasing, as pyinterp requires, and explicitly NOT
    periodic, which is the right geometry for a single KaRIn swath (the
    historic Axis on raw longitudes could be non-monotonic for wrapped
    segments, and is_circle=True wrongly interpolates across the 0/360 seam).
    The value columns are permuted to match the sorted axis and restored
    afterwards, so the returned array stays aligned with the caller's
    coordinates whatever the input ordering.

    Parameters
    ------------
    values : np.ndarray
        2D array (num_lines, num_pixels) with NaNs to fill.
    longitudes, latitudes : np.ndarray
        2D coordinate arrays of the swath segment.
    epsilon : float, optional
        Convergence threshold (used on the pyinterp >= 2026 path, matching the
        historic engine default), default 1e-4.
    num_threads : int, optional
        passed to pyinterp, default 16.

    Returns
    ---------
    has_converged : bool
        True when the fill converged (pyinterp >= 2026: final residual <= epsilon).
    filled_values : np.ndarray
        Filled array, same column order as the input.
    """
    x_values = np.mod(longitudes[0, :], 360.0)
    sort_idx = np.argsort(x_values)
    x_sorted = x_values[sort_idx]
    restore_idx = np.argsort(sort_idx)
    sorted_values = np.asarray(values, dtype="f8")[:, sort_idx]
    y_values = latitudes[:, 0]

    if _gauss_seidel_inplace_api():
        # pyinterp >= 2026: plain array, filled in place, (iterations, residual)
        _iterations, residual = fill.gauss_seidel(
            sorted_values,
            epsilon=epsilon,
            is_periodic=False,
            num_threads=num_threads,
        )
        has_converged = bool(residual <= epsilon)
        filled_values = sorted_values
    else:
        # legacy: Grid2D-backed (returns (has_converged, filled_data))
        x_axis = pyinterp.core.Axis(x_sorted, is_circle=False)
        y_axis = pyinterp.core.Axis(y_values, is_circle=False)
        grid = pyinterp.Grid2D(y_axis, x_axis, sorted_values)
        has_converged, filled_values = fill.gauss_seidel(grid, num_threads=num_threads)
        has_converged = bool(has_converged)

    return has_converged, filled_values[:, restore_idx]


def fill_nan(datasets, varname: str = "ssha"):
    """
    Fills in missing values (NaN) in each xarray.Dataset using Gauss-Seidel method.

    Works with both pyinterp generations: pyinterp >= 2026 (in-place
    gauss_seidel on a plain array) and the legacy Grid2D-based API.

    Parameters
    ------------
    datasets: Dict
        Dictionary containing xarray.Datasets
    varname: str, optional
        Variable name to fill in missing values.
        Defaults to "ssha"

    Returns
    ---------
    has_converged: bool
        Indicates whether the method has converged, returns True if the method has converged, otherwise returns False
    filled_dataset: Dict
        Dictionary containing xarray.Datasets (with missing values filled in)
    """
    has_converged = True
    filled_datasets = {}
    new_key = 0
    for key in range(len(datasets)):

        # Selects only the variable of interest
        data = datasets[key][[varname]]

        latitudes = data["latitude"].values
        longitudes = data["longitude"].values
        values = data[varname].values

        if longitudes.size > 0 and latitudes.size > 0:

            # Replace all values in columns 33 and 34 with NaN (the two middle columns)
            if values.shape[1] >= 34:
                values[:, 33:35] = np.nan

            # Identify columns entirely NaN
            nan_columns = np.all(np.isnan(values), axis=0)

            # fill NaN data
            _has_converged, filled_values = _gauss_seidel_fill(values, longitudes, latitudes)
            has_converged &= _has_converged

            # Restore columns entirely NaN
            filled_values[:, nan_columns] = np.nan

            # Add filled values to the output dictionary
            datasets[key][varname] = (("num_lines", "num_pixels"), filled_values)
            filled_datasets[new_key] = datasets[key]
            new_key = new_key + 1
        else:
            print(f"Size of longitudes/latitudes is zero for dict number {key}")

    return has_converged, filled_datasets


# =============================================================================
# check_directory
# =============================================================================


def check_directory(database_path, start_date_str, end_date_str):
    """
    
    Scans the folders in the `database_path` directory, identifies the folders 
    containing NetCDF files whose dates are between `start_date_str` and `end_date_str`, 
    and returns a list of these folder names.

    
    Parameters
    ------------
    database_path : str
        Path to the `database` directory
    start_date_str : str
        Start date in 'YYYYMMDD' format
    end_date_str : str
        End date in 'YYYYMMDD' format
    
    Returns
    ---------
    
    matching_folders : list
        List of folder names containing NetCDF files within the specified date range
        If an error occurs, an error message is printed and an empty list is returned.
        
    """

    # Regex pattern to match folder names like cycle_001, cycle_002, etc.
    folder_pattern = re.compile(r'cycle_\d{3}')
    # Regex pattern to match the date in the file name SWOT........_YYYYMMDD...
    file_pattern = re.compile(r'SWOT.*_(\d{8})T.*\.nc')

    # Convert date strings to datetime objects
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')

    # List to store the names of folders that meet the criteria
    matching_folders = []

    try:
        # List all items in the database directory
        items = os.listdir(database_path)

        for item in items:
            folder_path = os.path.join(database_path, item)

            # Check if the item is a directory and matches the pattern
            if os.path.isdir(folder_path) and folder_pattern.match(item):
                netcdf_files = [f for f in os.listdir(folder_path) if f.endswith('.nc')]
                for nc_file in netcdf_files:
                    match = file_pattern.search(nc_file)
                    if match:
                        file_date_str = match.group(1)
                        file_date = datetime.strptime(file_date_str, '%Y%m%d')

                        if start_date <= file_date <= end_date:
                            matching_folders.append(item)
                            break  # Stop checking files in this folder once a match is found

        return matching_folders

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# =============================================================================
# extract_xarrays_by_time
# =============================================================================


def extract_xarrays_by_time(database_path, start_date_str, end_date_str, area):
    """
    Processes folders in the `database_path` directory, applies the `extract_xarray_in_region` 
    function to each folder that contains NetCDF files within the date range specified 
    by `start_date_str` and `end_date_str`, and combines the results into a single dictionary.


    Parameters
    ------------
    database_path : str
        Path to the `database` directory
    start_date_str : str
        Start date in 'YYYYMMDD' format
    end_date_str : str
        End date in 'YYYYMMDD' format
    area : list
        List with the boundaries of the region of interest [longitude_min, latitude_min, longitude_max, latitude_max]
        
        
    Returns
    ---------
    
    combined_datasets_dict : Dict
        A dictionary of xarray.Datasets combining the results from `extract_xarray_in_region` function for each folder.
        
    """

    matching_folders = check_directory(database_path, start_date_str, end_date_str)
    combined_datasets_dict = defaultdict(list)
    current_key = 0

    for folder in matching_folders:
        folder_path = os.path.join(database_path, folder)
        result_dict = extract_xarray_in_region(folder_path, area)

        # Add entries to the combined_dict with sequential keys
        for value in result_dict.values():
            combined_datasets_dict[current_key] = value
            current_key += 1

    # Convert defaultdict back to dict
    combined_datasets_dict = dict(combined_datasets_dict)

    return combined_datasets_dict


# =============================================================================
# plot_obs_count
# =============================================================================


def plot_obs_count(obs_count, area, obs_count2=None, title=None, title2=None, save_fig=None):
    """
    Plots the number of observations on a geographical map
    
    Parameters
    ------------
    obs_count : numpy.ndarray
        A 2D array containing the count of observations in each geographical bin.  
    obs_count2 : numpy.ndarray, optional
        A second 2D array containing the count of observations in each geographical bin. 
    area : list
        List with the boundaries of the region of interest [longitude_min, latitude_min, longitude_max, latitude_max]     
    title : str, optional
        The title of the plot. Defaults to None.
    title2 : str, optional
        The title of the 2nd plot. Defaults to None.    
    save_fig: Optional[str], default None
        Name of the file to save the plot to.
        Does not save if None.

    
    Returns
    ---------
    None
        
    """
    lon_min, lat_min, lon_max, lat_max = area

    if lon_min > 180:
        lon_min = lon_min - 360

    if lon_max > 180:
        lon_max = lon_max - 360

    # Créer des sous-graphes côte à côte
    if obs_count2 is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        maxobs = max(obs_count.max(), obs_count2.max())
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 7), subplot_kw={'projection': ccrs.PlateCarree()})
        maxobs = obs_count.max()

    ax1.add_feature(cfeature.LAND, facecolor='gray')
    ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    # Create a mask for zero values to present the continent in its color
    obs_count_masked = np.ma.masked_where(obs_count == 0, obs_count)

    im1 = ax1.imshow(obs_count_masked, extent=[lon_min, lon_max, lat_min, lat_max],
                     origin='lower', cmap="jet", transform=ccrs.PlateCarree(),
                     vmin=0, vmax=maxobs)

    ax1.coastlines()
    gl1 = ax1.gridlines(draw_labels=True)
    gl1.right_labels = False
    gl1.top_labels = False

    plt.colorbar(im1, ax=ax1, label='Number of observations per bin', shrink=0.5)

    if title:
        if len(title) >= 40:
            ax1.set_title(title, fontsize=11, fontweight='bold', color='black')
        else:
            ax1.set_title(title, fontsize=15, fontweight='bold', color='black')

    if obs_count2 is not None:
        ax2.add_feature(cfeature.LAND, facecolor='gray')
        ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        # Create a mask for zero values to present the continent in its color
        obs_count_masked2 = np.ma.masked_where(obs_count2 == 0, obs_count2)

        im2 = ax2.imshow(obs_count_masked2, extent=[lon_min, lon_max, lat_min, lat_max],
                         origin='lower', cmap="jet", transform=ccrs.PlateCarree(),
                         vmin=0, vmax=maxobs)

        ax2.coastlines()
        gl2 = ax2.gridlines(draw_labels=True)
        gl2.right_labels = False
        gl2.top_labels = False
        plt.colorbar(im2, ax=ax2, label='Number of observations per bin', shrink=0.5)
        if title2:
            ax2.set_title(title2, fontsize=15, fontweight='bold', color='black')

    # Optionally save the figure
    if save_fig is not None:
        plt.savefig(save_fig)


# =============================================================================
# read_zarr_to_xarray_dict
# =============================================================================         


def read_zarr_to_xarray_dict(base_directory, area, start_date_str, end_date_str, variables_to_keep=None):
    """
    
    Reads Zarr files from a directory structure organized by month and day, converts them into a dictionnary of xarray.Dataset objects, retains only specified variables, and extracts a specific geographical region based on latitude and longitude limits.

    Parameters
    ------------
    base_directory : str
        The path to the base directory containing Zarr data organized by month and day.
    area : list
        List with the boundaries of the region of interest [longitude_min, latitude_min, longitude_max, latitude_max]
    start_date_str : str
        The desired start date in the format 'YYYYMMDD'.
    
    end_date_str : str
        The desired end date in the format 'YYYYMMDD'.
    
    variables_to_keep : list of str, optional
        A list of variable names to retain in each xarray.Dataset. If None, all variables are retained.
    
    Returns
    ---------
    datasets_dict : Dict
        A dictionary containing the resulting xarray.Dataset objects, indexed by unique integers.

    """

    lon_min = area[0]
    lon_max = area[2]
    lat_min = area[1]
    lat_max = area[3]

    # files number
    nfiles = 0
    datasets_dict = {}
    index = 0

    # Convert date strings into datetime objects
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')

    # Iterate through each day between the start and end dates
    current_date = start_date
    while current_date <= end_date:
        month = current_date.strftime('%m')
        day = current_date.strftime('%d')

        month_directory = os.path.join(base_directory, f'month={month}')
        #print(month_directory)
        if os.path.exists(month_directory):
            day_directory = os.path.join(month_directory, f'day={day}')
            #print(day_directory)
            if os.path.exists(day_directory):

                zarr_ds = xr.open_zarr(day_directory)

                # Keep only the specified variables if variables_to_keep is not None
                if variables_to_keep is not None:
                    zarr_ds = zarr_ds[variables_to_keep]

                coord_vars = ['latitude', 'longitude']
                zarr_ds = zarr_ds.set_coords(coord_vars)

                if lon_min > lon_max:

                    selection = (

                            (zarr_ds['latitude'] >= lat_min) &
                            (zarr_ds['latitude'] <= lat_max) &
                            (((zarr_ds['longitude'] >= lon_min) & (zarr_ds['longitude'] <= 360)) | (
                                        zarr_ds['longitude'] <= lon_max))
                    )
                else:
                    selection = (

                            (zarr_ds['latitude'] >= lat_min) &
                            (zarr_ds['latitude'] <= lat_max) &
                            (zarr_ds['longitude'] >= lon_min) &
                            (zarr_ds['longitude'] <= lon_max)
                    )

                selection = selection.compute()

                zarr_ds = zarr_ds.where(selection, drop=True)

                # create ssha variable
                zarr_ds['ssha'] = zarr_ds.duacs_ssha_karin_2_calibrated.where(zarr_ds.duacs_editing_flag == 0)
                zarr_ds = zarr_ds.drop_vars('duacs_ssha_karin_2_calibrated')
                zarr_ds = zarr_ds.drop_vars('duacs_editing_flag')

                # create mdt variable
                zarr_ds['mdt'] = zarr_ds.cvl_mean_dynamic_topography_cnes_cls_22
                zarr_ds = zarr_ds.drop_vars('cvl_mean_dynamic_topography_cnes_cls_22')

                datasets_dict[index] = zarr_ds

                index += 1
                nfiles += 1

            else:
                print(f'The directory for day{day} does not exist in month {month}')
        else:
            print(f'The directory for month{month} does not exist')

        # Move to the next day
        current_date += timedelta(days=1)

    return datasets_dict


# =============================================================================
# split_dsets_based_cnum
# =============================================================================  


def split_dsets_based_cnum(datasets_dict):
    """
    Splits xarray.dataset objects based on unique cycle and pass numbers.

    The function takes a dictionary of xarray.dataset objects and splits each dataset into smaller datasets based on unique values of 'cycle_number' and 'pass_number'. 
    The resulting datasets are stored in a new dictionary with sequential keys.
    
    Conditions:
    A dataset is split if it contains at least 2 different 'cycle_number' values.
    For each unique 'cycle_number', the dataset is further split if it contains at least 2 different 'pass_number' values.   
    If an xarray.dataset in the input dictionary meets the splitting conditions (having at least 2 different 'cycle_number' and 'pass_number'), it is split into smaller xarray datasets. Otherwise, the original dataset is included as is.

    Parameters
    -----------
    datasets_dict : Dict
        A dictionary where each key corresponds to an xarray Dataset.
        Each xarray Dataset is expected to have 'cycle_number' and 'pass_number' attributes.

    Returns
    --------
    splited_dict : Dict
        A new dictionary containing the split xarray.dataset objects.
    """

    splited_dict = {}
    index = 0

    for key in range(len(datasets_dict)):
        # Check if the condition is met(The datasets contain 2 cycle numbers.)
        if len(np.unique(datasets_dict[key].cycle_number.compute())) >= 2:
            # Retrieve the cycle number.
            long_cycle = len(np.unique(datasets_dict[key].cycle_number.compute()))

            for i in np.arange(long_cycle - 1):
                cycle = np.unique(datasets_dict[key].cycle_number.compute())[i]
                ds_cycle = datasets_dict[key].where(datasets_dict[key]['cycle_number'].compute() == cycle.astype(float),
                                                    drop=True)

                # Condition separation based on the pass numbers.
                if len(np.unique(ds_cycle.pass_number.compute())) >= 2:

                    long_pass = len(np.unique(ds_cycle.pass_number.compute()))

                    for i in np.arange(long_pass - 1):
                        passs = np.unique(ds_cycle.pass_number.compute())[i]
                        ds_cyclepass = ds_cycle.where(ds_cycle.pass_number.compute() == passs.astype(float), drop=True)

                        # Add the new datasets to the output dictionary with sequential keys
                        splited_dict[index] = ds_cyclepass
                        index += 1
        else:
            # If the dataset doesn't meet the condition, add it to the output dictionary as is
            splited_dict[index] = datasets_dict[key]
            index += 1

    return splited_dict


# =============================================================================
# _remove_duplicates_from_sys_path
# ============================================================================= 

def _remove_duplicates_from_sys_path():
    """
    Removes duplicates from the sys.path list while preserving the order of elements.

    Iterates through the sys.path list and constructs a new list without duplicates. 
    Updates sys.path with this new list.

    Returns
    --------
        None
        
    """
    seen = set()
    new_sys_path = []
    for path in sys.path:
        if path not in seen:
            new_sys_path.append(path)
            seen.add(path)
    sys.path = new_sys_path

    
# =============================================================================
# get_matching_cycles
# =============================================================================     


def get_matching_cycles(file_path, start_date_str, end_date_str):
    
    
    """
    Fetches a CSV file (local or S3) and finds cycle_numbers corresponding to a date range.
    
    Parameters
    ------------
    file_path : str
        Path to the file. Can be an S3 URL.
    start_date_str : str
        The desired start date in the format 'YYYYMMDD'.
    end_date_str : str
        The desired end date in the format 'YYYYMMDD'.
          
    Returns
    ---------
    cycle_numbers : list
        List of cycle_numbers that match the date range.
        
    """
    
    # Data cleaning: read every row without assuming a header, then identify
    # the data rows by coercing the time columns to datetimes. Junk preamble
    # lines and header rows (whose time cells do not parse) drop out, so the
    # function accepts cycle calendars with OR without leading junk lines
    # (ragged short junk rows are padded with NaN). The historic
    # implementation unconditionally discarded the first two rows, which
    # silently dropped the first two cycles on clean calendars.
    data = pd.read_csv(
        file_path, header=None,
        names=['cycle_number', 'start_time', 'end_time'],
    )
    data['cycle_number'] = data['cycle_number'].astype(str)
    data['start_time'] = pd.to_datetime(data['start_time'], errors='coerce')
    data['end_time'] = pd.to_datetime(data['end_time'], errors='coerce')
    data = data.dropna(subset=['start_time', 'end_time']).reset_index(drop=True)
    
    # Convert input dates to datetime
    start_date = datetime.strptime(start_date_str, "%d%m%Y")
    end_date = datetime.strptime(end_date_str, "%d%m%Y")
    
    # Filter for matching cycles
    matching_cycles = data[
        (data['start_time'] <= end_date) & (data['end_time'] >= start_date)
    ]['cycle_number'].tolist()
    
    return matching_cycles


# =============================================================================
# read_swot_ncfiles_S3folder
# =============================================================================

def read_swot_ncfiles_S3folder(s3_folder, endpoint_url, area, engine="h5netcdf"):
    """
    Load NetCDF files from S3 and filter them based on the region of interest.
    
    Parameters
    -----------
    s3_folder : str
        Path to the S3 folder containing NetCDF files.
    endpoint_url : str
        URL of the S3 endpoint.
    area : list
        List with the boundaries of the region of interest [lon_min, lat_min, lon_max, lat_max].
    engine : str, optional
        Engine for reading NetCDF files, default is "h5netcdf".
        
    Returns
    --------
    datasets_dict : Dict
        A dictionary of xarray datasets.
    """
    lon_min, lat_min, lon_max, lat_max = area
    
    # Initialize S3 filesystem
    fs = s3fs.S3FileSystem(anon=True, endpoint_url=endpoint_url)
    
    # List NetCDF files in the folder
    file_list = [file for file in fs.ls(s3_folder) if file.endswith('.nc')]
    
    # Initialize the output dictionary
    datasets_dict = {}
    current_key = 0
    
    for file in file_list:
        try:
            # Open the NetCDF file
            with fs.open(file, mode='rb') as fileObj:
                ds = xr.open_dataset(fileObj, engine=engine)
                
                #drop some variables
                ds = ds.drop_vars(["i_num_line", "i_num_pixel"], errors="ignore")
                    
                # Check geographical filtering
                if 'latitude' in ds and 'longitude' in ds:

                    # Handle longitude wrapping
                    if lon_min < lon_max:
                        lon_selection = (ds['longitude'] >= lon_min) & (ds['longitude'] <= lon_max)
                    else:
                        lon_selection = ((ds['longitude'] >= lon_min) & (ds['longitude'] <= 360)) | (ds['longitude'] <= lon_max)
                    
                    lat_selection = (ds['latitude'] >= lat_min) & (ds['latitude'] <= lat_max)
                    
                    # Combine the selection masks
                    selection = lat_selection & lon_selection
                    
                    # Check if selection is valid
                    if selection.any():
                        print(f"{file[61:68]} included.")
                    else :    
                        ds.close()
                        continue
                    
                    # Drop data outside the region
                    ds = ds.where(selection, drop=True)
                    
                    # Check if the filtered dataset has valid data
                    if ds['latitude'].size == 0 or ds['longitude'].size == 0:
                        print(f"File {file} excluded: empty dataset after region filtering.")
                        ds.close()
                        continue
                    
                # Add the dataset to the dictionary if it passed all filters
                datasets_dict[current_key] = ds
                current_key += 1
                
        except Exception as e:
            print(f"Error processing file {file[61:68]}: {e}")
            continue
    
    return datasets_dict


# =============================================================================
# read_swot_ncfiles_S3subfolders
# =============================================================================

def read_swot_ncfiles_S3subfolders(base_s3_folder, cycle_numbers, endpoint_url, area, engine="h5netcdf"):
    """
    Load NetCDF files from multiple S3 subfolders and store them in a dictionary.
    
    Parameters
    -----------
    base_s3_folder : str
        Base path to the S3 folder containing subfolders for each cycle.
    cycle_numbers : list
        List of specific cycle numbers (subfolders) to process (e.g., [500, 502, 501]).
    endpoint_url : str
        URL of the S3 endpoint.
    area : list
        List with the boundaries of the region of interest [lon_min, lat_min, lon_max, lat_max].
    engine : str, optional
        Engine for reading NetCDF files, default is "h5netcdf".
        
    Returns
    -------
    datasets_dict : Dict
        A dictionary of xarray datasets.
    """
    datasets_dict = {}
    current_key = 0

    for cycle_number in cycle_numbers:
        s3_folder = f"{base_s3_folder}/cycle_{cycle_number}/"
        
        try:
            # Use the previous function to load datasets from the specific folder
            datasets_from_cycle = read_swot_ncfiles_S3folder(
                s3_folder=s3_folder,
                endpoint_url=endpoint_url,
                area=area,
                engine=engine
            )
            
            # Add the datasets from this cycle to the dictionary
            for ds in datasets_from_cycle.values():
                datasets_dict[current_key] = ds
                current_key += 1
        
        except Exception as e:
            print(f"Error processing cycle {cycle_number}: {e}")
            continue
    
    return datasets_dict

# =============================================================================
# sort_datasets_by_time
# =============================================================================

def sort_datasets_by_time(datasets_dict):
    """
    Sorts a dictionary of xarray datasets by the `time_coverage_begin` attribute.
    
    Parameters
    -----------
    datasets_dict : dict
        Dictionary where keys are integers and values are xarray datasets.
    
    Returns
    -------
    sorted_datasets : dict
        A new dictionary with datasets sorted by their `time_coverage_begin` attribute.
    """
    # Step 1: Extract and convert time_coverage_begin to datetime
    dataset_times = []
    for key, dataset in datasets_dict.items():
        try:
            # Assume dataset.time_coverage_begin is a string
            time_str = str(dataset.time_coverage_begin)
            # Convert to datetime object
            time_dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            dataset_times.append((key, time_dt, dataset))
        except Exception as e:
            print(f"Skipping dataset {key}: invalid time_coverage_begin format. Error: {e}")
            continue
    
    # Step 2: Sort datasets by their datetime
    dataset_times_sorted = sorted(dataset_times, key=lambda x: x[1])
    
    # Step 3: Reorganize the dictionary with reindexed keys
    sorted_datasets = {i: item[2] for i, item in enumerate(dataset_times_sorted)}
    
    return sorted_datasets


# =============================================================================
# generate_plots
# =============================================================================

def generate_plots(sorted_datasets, area, output_dir):
    """
    Generate and save superposed plots for datasets in a dictionary.
    
    Parameters
    -----------
    sorted_datasets : Dict
        Dictionary of datasets, where each key corresponds to a dataset.
    area : list
        List specifying the boundaries of the region [lon_min, lat_min, lon_max, lat_max].
    output_dir : str
        Directory to save the plots.
        
    Returns
    -------
    None
    """
    # Extract boundaries from the area list
    lon_min, lat_min, lon_max, lat_max = area
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot configuration
    plot_kwargs = dict(
        x="longitude",
        y="latitude",
        cmap="Spectral_r",
        vmin=-0.3,
        vmax=0.3,
    )
    
    # Initialize the global figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), subplot_kw=dict(projection=ccrs.PlateCarree()))
    ax.coastlines()
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=True)
    ax.add_feature(cfeature.LAND, facecolor='grey')
    
    # Loop over the datasets
    for key, dataset in sorted_datasets.items():
        # Adjust colorbar settings
        current_plot_kwargs = plot_kwargs.copy()
        if key == 0:  # Add colorbar only for the first plot
            current_plot_kwargs["add_colorbar"] = True
            current_plot_kwargs["cbar_kwargs"] = {"shrink": 0.5}
        else:  # Disable colorbar for subsequent plots
            current_plot_kwargs["add_colorbar"] = False

        # Add the current dataset to the plot
        dataset.ssha.plot.pcolormesh(ax=ax, **current_plot_kwargs)
        
        # Set the title
        ax.set_title(str(dataset.time_coverage_begin))
        
        # Save the plot after each overlay
        output_path = os.path.join(output_dir, f"plot_{key}.png")
        plt.savefig(output_path, dpi=300)

    print(f"Plots saved in '{output_dir}'.")
    
    
# =============================================================================
# make_movie_swot
# =============================================================================
 
def make_movie_swot(image_folder, output_video, fps=3):
    """
    Create a video from images.

    Parameters
    -----------
    image_folder : str
        Folder containing the images to be used in the video.
    output_video : str
        Name of the output video file.
    fps : int, optional
        Frames per second for the video, default is 3.

    Returns
    -------
    None
    """
    # Get the list of image files, sorted naturally
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = natsorted(images)  # Natural sorting (e.g., 1, 2, 10 instead of 1, 10, 2)

    # Check if there are images available
    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to determine video dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Add each image to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release resources
    video.release()
    cv2.destroyAllWindows()

    print(f"Video saved as '{output_video}'")
    

# =============================================================================
# _parse_swot_filename
# =============================================================================

def _parse_swot_filename(filename, index_name="SSH"):
    """
    Extract (pass_number, end_date) from a SWOT-like NetCDF filename.

    Two conventions are tried in order:

    1. A regex matching names of the form
       ``*_<index_name>_<cycle>_<pass>_<start><anything>_<end><anything>*.nc``
       (covers the SWOT-OMIP pseudo-SWOT layout, e.g.
       ``MODEL_GRID_L3_LR_SSH_011_532_20240215000000_20240225235929_v1.0.2.nc``).
    2. The token-based fallback
       ``*_<index_name>_<cycle>_<pass>_<start>_<end>*
       (the pass number is the 2nd token after ``index_name``, the end date the
       4th) — same tokens, kept for names the regex does not cover, e.g.
       ``SWOT_L3_LR_SSH_Expert_532_011_20240215T..._20240217T....nc``.

    Parameters
    ------------
    filename : str
        Name (or path) of the file.
    index_name : str, optional
        Filename token identifying the data index, default "SSH".

    Returns
    ---------
    (pass_number, end_date) : tuple(int, str) or None
        pass_number as int, end_date as 'YYYYMMDD'. None when the name does
        not follow either convention (unlike the historic parts.index() parsing,
        which raised ValueError mid-scan).
    """
    match = re.search(
        rf"_{re.escape(index_name)}_(\d+)_(\d+)_(\d{{8}})[^_]*_(\d{{8}})[^_]*",
        filename,
    )
    if match:
        return int(match.group(2)), match.group(4)[:8]
    parts = filename.split("_")
    try:
        idx = parts.index(index_name)
        return int(parts[idx + 2]), parts[idx + 4][:8]
    except (ValueError, IndexError):
        return None


# =============================================================================
# read_swot_ncfiles_local_list
# =============================================================================

def read_swot_ncfiles_local_list(folder, area, file_list, engine="h5netcdf"):
    """
    Load local NetCDF files and filter them based on the region of interest.
    Same reader as read_swot_ncfiles_S3folder, but for a precomputed list of
    local file paths (e.g. the output of find_listdata_local).

    Parameters
    -----------
    folder : str
        Local folder containing the NetCDF files (informational; files are
        taken from file_list).
    area : list
        List with the boundaries of the region of interest [lon_min, lat_min, lon_max, lat_max].
    file_list : list
        List of file paths to read.
    engine : str, optional
        Engine for reading NetCDF files, default is "h5netcdf".

    Returns
    --------
    datasets_dict : Dict
        A dictionary of xarray datasets.
    """
    lon_min, lat_min, lon_max, lat_max = area

    datasets_dict = {}
    current_key = 0

    for file in file_list:
        try:
            # Open the NetCDF file
            with open(file, mode='rb') as fileObj:
                ds = xr.open_dataset(fileObj, engine=engine)

                # drop some variables
                ds = ds.drop_vars(["i_num_line", "i_num_pixel"], errors="ignore")

                # Check geographical filtering
                if 'latitude' in ds and 'longitude' in ds:

                    # Handle longitude wrapping
                    if lon_min < lon_max:
                        lon_selection = (ds['longitude'] >= lon_min) & (ds['longitude'] <= lon_max)
                    else:
                        lon_selection = ((ds['longitude'] >= lon_min) & (ds['longitude'] <= 360)) | (ds['longitude'] <= lon_max)

                    lat_selection = (ds['latitude'] >= lat_min) & (ds['latitude'] <= lat_max)

                    # Combine the selection masks
                    selection = lat_selection & lon_selection

                    # Check if selection is valid
                    if selection.any():
                        logger.info("%s included.", os.path.basename(file))
                    else:
                        ds.close()
                        continue

                    # Drop data outside the region
                    ds = ds.where(selection, drop=True)

                    # Check if the filtered dataset has valid data
                    if ds['latitude'].size == 0 or ds['longitude'].size == 0:
                        logger.info("File %s excluded: empty dataset after region filtering.",
                                    os.path.basename(file))
                        ds.close()
                        continue

                # Add the dataset to the dictionary if it passed all filters
                datasets_dict[current_key] = ds
                current_key += 1

        except Exception as e:
            logger.warning("Error processing file %s: %s", os.path.basename(file), e)
            continue

    return datasets_dict


# =============================================================================
# read_swot_ncfiles_S3folder_list
# =============================================================================

def read_swot_ncfiles_S3folder_list(s3_folder, endpoint_url, area, file_list, engine="h5netcdf"):
    """
    Load NetCDF files from S3 and filter them based on the region of interest.
    Same reader as read_swot_ncfiles_S3folder, but for a precomputed list of S3
    keys (e.g. the output of find_listdata) instead of listing s3_folder.

    Parameters
    -----------
    s3_folder : str
        Path to the S3 folder containing the NetCDF files (informational; files
        are taken from file_list).
    endpoint_url : str
        URL of the S3 endpoint.
    area : list
        List with the boundaries of the region of interest [lon_min, lat_min, lon_max, lat_max].
    file_list : list
        List of S3 keys to read.
    engine : str, optional
        Engine for reading NetCDF files, default is "h5netcdf".

    Returns
    --------
    datasets_dict : Dict
        A dictionary of xarray datasets.
    """
    lon_min, lat_min, lon_max, lat_max = area

    # Initialize S3 filesystem
    fs = s3fs.S3FileSystem(anon=True, endpoint_url=endpoint_url)

    datasets_dict = {}
    current_key = 0

    for file in file_list:
        try:
            # Open the NetCDF file
            with fs.open(file, mode='rb') as fileObj:
                ds = xr.open_dataset(fileObj, engine=engine)

                # drop some variables
                ds = ds.drop_vars(["i_num_line", "i_num_pixel"], errors="ignore")

                # Check geographical filtering
                if 'latitude' in ds and 'longitude' in ds:

                    # Handle longitude wrapping
                    if lon_min < lon_max:
                        lon_selection = (ds['longitude'] >= lon_min) & (ds['longitude'] <= lon_max)
                    else:
                        lon_selection = ((ds['longitude'] >= lon_min) & (ds['longitude'] <= 360)) | (ds['longitude'] <= lon_max)

                    lat_selection = (ds['latitude'] >= lat_min) & (ds['latitude'] <= lat_max)

                    # Combine the selection masks
                    selection = lat_selection & lon_selection

                    # Check if selection is valid
                    if selection.any():
                        logger.info("%s included.", file)
                    else:
                        ds.close()
                        continue

                    # Drop data outside the region
                    ds = ds.where(selection, drop=True)

                    # Check if the filtered dataset has valid data
                    if ds['latitude'].size == 0 or ds['longitude'].size == 0:
                        logger.info("File %s excluded: empty dataset after region filtering.", file)
                        ds.close()
                        continue

                # Add the dataset to the dictionary if it passed all filters
                datasets_dict[current_key] = ds
                current_key += 1

        except Exception as e:
            logger.warning("Error processing file %s: %s", file, e)
            continue

    return datasets_dict


# =============================================================================
# find_listdata
# =============================================================================

def find_listdata(matching_cycles, endpoint_url, bucket, pass_numbers, start_date, end_date, index_name="SSH"):
    """
    Find .nc files in S3 matching cycles, pass numbers, and months.

    Parameters:
        matching_cycles (list[int]): List of cycle numbers to search.
        endpoint_url (str): URL of the S3 endpoint.
        bucket (str): Full base folder path of the dataset, including the
            's3://' scheme and bucket name (e.g. 's3://project-moi-swot-omip/GLO12V4').
            Cycle folders are globbed as '{bucket}/cycle_{n}/**/*.nc', so a bare
            bucket name is not enough.
        pass_numbers (list[int]): List of allowed pass numbers.
        start_date (str): Start date in format 'ddmmyyyy'.
        end_date (str): End date in format 'ddmmyyyy'.

    Returns:
        list_files (list[str]): List of S3 keys that match the filters.
    """
    # Convert start/end dates to month integers
    m1 = datetime.strptime(start_date, "%d%m%Y").month
    m2 = datetime.strptime(end_date, "%d%m%Y").month

    # Initialize S3 filesystem USING S3FS
    fs = s3fs.S3FileSystem(anon=True, endpoint_url=endpoint_url)  # assumes AWS credentials are set in environment or ~/.aws/credentials
    list_files = []

    for cycle_number in matching_cycles:
        # Construct the S3 folder path
        s3_folder = f"{bucket}/cycle_{cycle_number}/"

        # Recursively find all .nc files
        all_files = fs.glob(s3_folder + "**/*.nc")  # ** for recursive search

        for filename in all_files:
            parsed = _parse_swot_filename(filename, index_name)
            if parsed is None:
                logger.warning("Skipping %s: unrecognized filename.", filename)
                continue
            pass_number, file_end_date = parsed
            month = datetime.strptime(file_end_date, "%Y%m%d").month
            if (month in range(m1, m2 + 1)) & (pass_number in pass_numbers):
                logger.debug("%s", filename)
                list_files.append(filename)

    logger.info("%d files found", len(list_files))
    return list_files


# =============================================================================
# find_listdata_local
# =============================================================================

def find_listdata_local(matching_cycles, folder, pass_numbers, start_date, end_date, index_name="SSH"):
    """
    Find local .nc files matching cycles, pass numbers, and months.
    Local-filesystem twin of find_listdata: cycle folders are globbed as
    '{folder}/cycle_{n}/*.nc'.

    Parameters:
        matching_cycles (list[int]): List of cycle numbers to search.
        folder (str): Local folder containing the cycle_XXX subfolders.
        pass_numbers (list[int]): List of allowed pass numbers.
        start_date (str): Start date in format 'ddmmyyyy'.
        end_date (str): End date in format 'ddmmyyyy'.

    Returns:
        list_files (list[str]): List of file paths that match the filters.
    """
    # Convert start/end dates to month integers
    m1 = datetime.strptime(start_date, "%d%m%Y").month
    m2 = datetime.strptime(end_date, "%d%m%Y").month

    list_files = []

    for cycle_number in matching_cycles:
        # Recursively find all .nc files
        all_files = glob.glob(folder + f"/cycle_{cycle_number}/*.nc")

        for filename in all_files:
            parsed = _parse_swot_filename(filename, index_name)
            if parsed is None:
                logger.warning("Skipping %s: unrecognized filename.", filename)
                continue
            pass_number, file_end_date = parsed
            month = datetime.strptime(file_end_date, "%Y%m%d").month
            if (month in range(m1, m2 + 1)) & (pass_number in pass_numbers):
                logger.debug("%s", filename)
                list_files.append(filename)

    logger.info("%d files found", len(list_files))
    return list_files


# =============================================================================
# normalize_longitude
# =============================================================================

def normalize_longitude(geometry):
    """Normalize longitude coordinates to [-180, 180] range.

    Points are returned as shapely Points; polygons are rebuilt with
    normalized exterior and interior rings (holes are preserved — note that
    assigning to Polygon.interiors, as done historically, is a no-op on
    shapely 2). Other geometry types are returned unchanged.
    """
    if geometry.geom_type == 'Point':
        x, y = geometry.x, geometry.y
        x_norm = ((x + 180) % 360) - 180
        return Point(x_norm, y)
    elif hasattr(geometry, 'exterior'):
        # For polygons, transform exterior and interior coordinates
        def transform_coords(coords):
            return [((lon + 180) % 360 - 180, lat) for lon, lat in coords]

        exterior = geometry.exterior
        coords = transform_coords(list(exterior.coords))
        interiors = [transform_coords(list(interior.coords)) for interior in geometry.interiors]
        return Polygon(coords, holes=interiors)
    return geometry


# =============================================================================
# get_half_orbits_intersect
# =============================================================================

def get_half_orbits_intersect(bbox, GEOMETRIES_FILE):
    """Get half orbits that intersect a bounding box.

    Parameters
    ----------
    bbox:
        the bounding box

    Returns
    -------
     gpd.GeoDataFrame:
        A Geopandas dataframe containing intersecting half orbits numbers and geometries
    """
    import geopandas as gpd  # deferred so that importing this module does not require geopandas

    swath_geometries = gpd.read_file(GEOMETRIES_FILE)

    bbox_polygon = geometry.box(*bbox)

    def _filter_intersect(row, polygon):
        half_orbit_polygon = row.geometry
        return polygon.intersects(half_orbit_polygon)

    select = swath_geometries.apply(_filter_intersect, polygon=bbox_polygon, axis=1)
    return swath_geometries[select]


# =============================================================================
# swath_search
# =============================================================================

def swath_search(lsw, lNE, latsw, latNE, GEOMETRIES_FILE):
    """Return the pass numbers of the half orbits intersecting a lat/lon box."""
    lon_range = lsw, lNE
    lat_range = latsw, latNE
    bbox = [lon_range[0], lat_range[0], lon_range[1], lat_range[1]]
    swath_geoms = get_half_orbits_intersect(bbox, GEOMETRIES_FILE)
    half_orbits = list(swath_geoms['pass_number'])

    return np.transpose(half_orbits)


# =============================================================================
# make_circular_axis
# =============================================================================

def make_circular_axis(lon_array):
    """
    Return a strictly-increasing longitude array in [0, 360).

    Longitudes are taken modulo 360 and sorted. No pyinterp object is built
    here (the name is kept for backwards compatibility with existing call
    sites); pass the result to a pyinterp Axis with is_circle=False.
    """
    # Ensure all values in [0, 360)
    lon = np.mod(lon_array, 360)

    # Sort to make strictly increasing
    sort_idx = np.argsort(lon)
    lon_sorted = lon[sort_idx]

    return lon_sorted
    