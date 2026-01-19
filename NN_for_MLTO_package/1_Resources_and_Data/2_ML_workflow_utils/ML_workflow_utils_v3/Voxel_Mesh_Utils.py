"""

This script contains functions that are used for various tasks within the training workflow. See individual descriptions.

"""

import numpy as np
import copy
import pandas as pd
import plotly.express as px

def target_binarray_threshold(array, volfrac_target, cubic_shape=64):
    target_num_elements = round(volfrac_target*(cubic_shape**3))
    
    array_sort = np.sort(array.reshape((-1,)))[::-1]
    
    threshold  = array_sort[target_num_elements]
    
    binarray = (array >= threshold).astype(int)
    
    volfrac = (np.sum(binarray == 1)) / (cubic_shape**3)
    
    return binarray, threshold, volfrac
    
    

def Plot_Array(array, colorscale='sunset', width=800, height=800, marker_scaler=8, marker_size=10, x_range=None, y_range=None, z_range=None, bincolor='black', lincolor='darkgrey',
                                plotpath = None, show=False, export_png=True, export_html=False, binary=False, scale_markers=False, title=None, symbol='circle'):
    """
    Plot Voxel arrays, either continuous or binary

    Args:
        array (np.ndarray): 3D numpy array, representing voxel model
        colorscale (str, optional): plotly color scales. Defaults to 'sunset'. See https://plotly.com/python/builtin-colorscales/ for options.
        width (int, optional): Plot width. Defaults to 800.
        height (int, optional): Plot height. Defaults to 800.
        marker_scaler (int, optional): For continuous arrays, scales the markers based on their values. Defaults to 8.
        marker_size (int, optional): For binary arrays, or if a constant marker size is desired for continuous arrays, sets the marker size. Defaults to 10.
        x_range (tuple, optional): Range of x-axis to plot. Defaults to None.
        y_range (tuple, optional): Range of y-axis to plot. Defaults to None.
        z_range (tuple, optional): Range of z-axis to plot. Defaults to None.
        bincolor (str, optional): Color of the markers for binary plotting. Defaults to 'black'. See https://community.plotly.com/t/plotly-colours-list/11730/3 for lists of colors.
        lincolor (str, optional): Color of the border of the markers for binary plotting. Defaults to 'darkgrey'. See https://community.plotly.com/t/plotly-colours-list/11730/3 for lists of colors.
        plotpath (str, optional): Path for saving the plot either as a .png or (interactive) .html. Defaults to None.
        show (bool, optional): Flag for showing the plot. Defaults to False.
        export_png (bool, optional): Flag for saving the plot as a .png file. Defaults to True.
        export_html (bool, optional): Flag for saving the plot as an interactive .html page. Defaults to False.
        binary (bool, optional): Flag for indicating that the input array is a binary voxel array. Defaults to False.
        scale_markers (bool, optional): Flag for scaling the markers for a continuous array. Defaults to False.
        title (str, optional): Title of plot. Defaults to None.
        symbol (str, optional): Symbol for the markers. Defaults to 'circle'. See https://plotly.com/python/marker-style/ for list of symbols.
    """
    
    z, x, y = array.nonzero()
    
    # Filter data based on specified ranges
    if x_range is not None:
        x_mask = (x >= x_range[0]) & (x <= x_range[1])
        x = x[x_mask]
        y = y[x_mask]
        z = z[x_mask]
        
    if y_range is not None:
        y_mask = (y >= y_range[0]) & (y <= y_range[1])
        x = x[y_mask]
        y = y[y_mask]
        z = z[y_mask]
        
    if z_range is not None:
        z_mask = (z >= z_range[0]) & (z <= z_range[1])
        x = x[z_mask]
        y = y[z_mask]
        z = z[z_mask]

    
    arr_plot_df = pd.DataFrame({'x': x, 'y': y, 'z': z})
    

    if binary:
        p = px.scatter_3d(x=arr_plot_df['x'], y=arr_plot_df['y'], z=arr_plot_df['z'], width=width, height=height, title=title)
        p.update_traces(marker=dict(size=marker_size, color=bincolor, line=dict(width=0.1, color=lincolor), symbol=symbol))

        # p.update_traces(marker_size=3)                                        
        
    else:
        value_list = []
        for i in range(len(x)):
            value_list.append(array[z[i], x[i], y[i]])

        arr_plot_df["activation_value"] = value_list
        
        if scale_markers:
            arr_plot_df['markersize_scaler'] = (arr_plot_df["activation_value"] - arr_plot_df["activation_value"].min()) / (arr_plot_df["activation_value"].max() - arr_plot_df["activation_value"].min())
        else:
            pass

        p = px.scatter_3d(arr_plot_df,
                          x='x', 
                          y='y', 
                          z='z', 
                          color='activation_value', 
                          opacity=1.0,
                          color_continuous_scale=colorscale,
                          width=width,
                          height=height,
                          )

        p.update_traces(marker=dict(size=12, line=dict(width=0.1, color=lincolor), symbol=symbol))
        if scale_markers:
            p.update_traces(marker_size=arr_plot_df['markersize_scaler'] * marker_scaler)
        else:
            p.update_traces(marker_size=marker_size)
    
        
    
    if show:
        p.show()
    else:
        pass
    
    if export_png:
        p.write_image(plotpath+'.png')
    else:
        pass
    
    if export_html:
        p.write_image(plotpath+'.html')
    else:
        pass
    del p, arr_plot_df



def UnScaleData(csv, param, value, scaled='minmax'):
    if scaled=='minmax':
        max = csv[param].max()
        min = csv[param].min()
        unscaled = value * (max-min) + min
    elif scaled=='normalized':
        mean = csv[param].mean()
        std = csv[param].std()
        unscaled = value * std + mean
    return unscaled


def PaddingWraparound(tensor, axis=(0,1,2), padding=(2,2,2), as_numpy=True):
    """
        Adds padding to tensor (array) of specified size, where the value of the padding is from the opposite
        extent of the input tensor, resulting in 'wraparound' or 'periodic' padding.

        In the case of unit cells, this type of padding simulates ingesting a unit cell that is part of an 
        indefinitely tessellated array of unit cells, such that the padding is drawing from the neighboring
        unit cells.

        Example:

        Original,                       padded by 1 unit on each side along the y (column) axis:
        [[0, 1, 2],                          [[2, 0, 1, 2, 0],  
         [3, 4, 5],                           [5, 3, 4, 5, 3],
         [6, 7, 8]]                           [8, 6, 7, 8, 6,]]   

    Args:
        tensor (np.ndarray): input 3D array of unit cell - size (64,64,64) in this package's application
        axis (tuple, optional): axes along which to pad. Defaults to (0,1,2). For this package's purposes,
            all unit cells are padded in all dimensions.
        padding (int or tuple, optional): number of cells to pad. Defaults to (2,2,2).
            For this package, cells are padded as follows - 2 (padding) + 64 (original) + 2 (padding) = 68.
            These original unit cells feed into the autoencoder. Its first convolutional layer reduces dimensions from 
            68 to 64, simulating computer vision on a volume slightly larger than the unit cell dimensions, in an
            indefinitely tessellated matrix of unit cells.
        as_numpy (bool, optional): converts tensor to numpy.ndarray. Defaults to True.

    Returns:
        tensor: the tensor with wraparound padding
    

        Adds padding of specified dimensions

        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: 

        return: padded tensor
    """


    if isinstance(axis,int):
        axis = (axis,)
    if isinstance(padding,int):
        padding = (padding,)

    ndim = len(tensor.shape)
    for ax,p in zip(axis,padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left

        ind_right = [slice(-p,None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right[0], ind_right[1], ind_right[2]]
        left = tensor[ind_left[0], ind_left[1], ind_left[2]]
        middle = tensor
        tensor = np.concatenate([right,middle,left], axis=ax)
    if as_numpy:
        return np.asarray(tensor)
    else:
        return tensor