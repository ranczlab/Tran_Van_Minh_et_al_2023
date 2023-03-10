import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys as sys
from mpl_toolkits.mplot3d import axes3d
from scipy.spatial import distance

import matplotlib.cm as cm
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder

# check distances between points to detect overlaps
def dist_to_overlap(pts, rad):
    distances = distance.cdist(pts, pts, 'euclidean')
    ddf = pd.DataFrame(distances)
    overlap = ddf[(ddf<2*rad) & (ddf != 0)].stack().reset_index() # cells overlap if distance (cellA - cellB) < 2 * cell radius
    overlap = pd.DataFrame(np.sort(overlap[['level_0','level_1']],1),index=overlap.index).drop_duplicates(keep='first') # drop the symmetrical duplicates
    return overlap


def check_overlap(overlap):
    if overlap.empty:
        print('empty')
    else:
        list_overlap = ['{} and {}'.format(i, j) for i, j in list(zip(overlap[0], overlap[1]))]
        print('overlapping spheres are :')
        for i in range(overlap.shape[0]):
            print('{} and {}'.format(overlap.iloc[i,0], overlap.iloc[i,1])) 

            
def points_to_drop(df):
    # from overlapping dataframe
    df_counts = df.stack().value_counts()
    df_multi = df_counts[df_counts > 1].index
    df_drop = df[~df.isin(df_multi)].dropna()
    to_drop = list(df_multi) + [int(i) for i in df_drop[0].values]
    return to_drop


def drop_from_points(pts, list_drop):
    return [i for j, i in enumerate(pts) if j not in list_drop]


def all_slices_pos(vol_side, thickn):
    q, r = divmod(vol_side, thickn)
    all_slices = []
    for i in range(q+1):
        all_slices.append(i * thickn)
    return all_slices


def kept_slices(list_slices, keep):
    return list_slices[0::keep]


def slicing_model_generate(seed, block_xy, block_z, num_points, radius):
    np.random.seed(seed)
    # create experimental data
    xrange = (0, block_xy)
    yrange = (0, block_xy)
    zrange = (0, block_z)
    points = []
    [ points.append((np.random.uniform(*xrange), np.random.uniform(*yrange), np.random.uniform(*zrange))) for i in range(num_points) ]
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    
    overlapping = dist_to_overlap(points, radius)
    drop_list = points_to_drop(overlapping)
    points = drop_from_points(points, drop_list)
    overlapping = dist_to_overlap(points, radius)
    
    # count cells in every kept slice 
    points_df = pd.DataFrame(points)
    
    return points_df
     
    
def slicing_model_count(points_df, radius, block_z, thickness, keep_every):
    all_slices_y = all_slices_pos(block_z, thickness)
    slices_y_keep = kept_slices(all_slices_y, keep_every)
    
    points_df = points_df.sort_values(by = [1])
    points_df['binned'] = pd.cut(points_df[1],all_slices_y)
    points_df['bins_left']= points_df["binned"].apply(lambda x: x.left)
    points_df['bins_right']= points_df["binned"].apply(lambda x: x.right)
    ord_enc = OrdinalEncoder()
    points_df["bins_code"] = ord_enc.fit_transform(points_df[['binned']])
    points_df['y_left_edge'] = points_df[1]-radius
    points_df['y_right_edge'] = points_df[1]+radius
    points_df['binned_left'] = pd.cut(points_df['y_left_edge'], all_slices_y)
    points_df['binned_right'] = pd.cut(points_df['y_right_edge'], all_slices_y)

    points_df['bins_left_left']= points_df["binned_left"].apply(lambda x: x.left)
    points_df['bins_left_right']= points_df["binned_left"].apply(lambda x: x.right)
    points_df['bins_right_left']= points_df["binned_right"].apply(lambda x: x.left)
    points_df['bins_right_right']= points_df["binned_right"].apply(lambda x: x.right)

    points_df['num_slices'] = points_df[['binned', 'binned_left', 'binned_right']].stack().groupby(level=0).nunique()
    points_kept = points_df[(points_df.bins_left_left.isin(slices_y_keep)) | (points_df.bins_right_left.isin(slices_y_keep))]

    if keep_every == 1:
        cells_per_slice = []
        for this_slice in points_df.binned.unique():
            cells_in_slice = points_df[(points_df.binned_left == this_slice) | (points_df.binned_right == this_slice)]
            this_slice_count = {'slice' : this_slice, 'cell_count' : len(cells_in_slice.index.unique()), 'cell_list' : cells_in_slice.index.unique()}
            cells_per_slice.append(this_slice_count)
        cells_per_slice_df = pd.DataFrame(cells_per_slice)
        cells_estimated = cells_per_slice_df.cell_count.sum() 
    else:
        cells_estimated = points_kept.shape[0] * keep_every
        
    cells_real = points_df.shape[0]
    cells_ratio = cells_estimated/cells_real
    
    return cells_real, cells_estimated, cells_ratio 
