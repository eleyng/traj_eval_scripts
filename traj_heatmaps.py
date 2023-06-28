### Purpose: to create heatmaps of state visitations for a given method.
### Method: records the states visited from each trajectory, and normalizes across total number of states from total number of trajecotries. Can specify different subdirectories to compare different trajectories collected from different baselines (i.e. bc-lstm-gmm, co-gail, etc.). 

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib import rcParams 
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib.pyplot as plt

# Place subdirectories containing trajectories in a list called 'main_directories'
main_directories = [
    '/home/eleyng/table-carrying-ai/results/hil-dp/Demonstrations',
    '/home/eleyng/table-carrying-ai/results/hil-dp/CoDP-H',
    '/home/eleyng/table-carrying-ai/results/hil-dp/CoDP',
    '/home/eleyng/table-carrying-ai/results/hil-dp/BC-LSTM-GMM',
    '/home/eleyng/table-carrying-ai/results/hil-dp/Co-GAIL',
    '/home/eleyng/table-carrying-ai/results/hil-dp/VRNN'
]

fig, axes = plt.subplots(1, 6, figsize=(15, 3), sharex=True, sharey=True)  # Create subplots with 1 row and 5 columns
plt.tight_layout()  # Adjust subplot spacing

# Loop through the main directories
for i, main_directory in enumerate(main_directories):
    # Initialize empty lists
    x_coords = []
    y_coords = []

    # Loop through subdirectories in the main directory
    for subdir in os.listdir(main_directory):
        subdir_path = os.path.join(main_directory, subdir)

        # Check if the item is a directory
        if os.path.isdir(subdir_path):

            # Loop through files in the subdirectory
            for file_path in glob.glob(os.path.join(subdir_path, '*.npz')):
                # Check if the file is a .npz file
                if file_path.endswith('.npz'):
                    # Load the .npz file
                    data = np.load(file_path)

                    # Extract the 'states' data
                    states = data['states']
                    x_coords.extend(states[:, 0])
                    y_coords.extend(states[:, 1])


    # Create a DataFrame with the extracted coordinates
    df = pd.DataFrame({'x': x_coords, 'y': y_coords})

    # Plot a heat map on the corresponding subplot
    heatmap = axes[i].hist2d(df['x'], df['y'], bins=[100,50], range=[[0,1200],[0, 600]], cmap='viridis', norm=mcolors.PowerNorm(vmin=0.01, vmax=800
    , gamma=0.45))
    # pcm = axes[i].hist2d(df['x'], df['y'], bins=[100,100], range=[[0,1200],[0, 600]],
    #                norm=mcolors.LogNorm(),
    #                cmap='viridis', shading='auto')
    # axes[i].set_xlabel('X Coordinates')
    # axes[i].set_ylabel('Y Coordinates') if i ==0 else None
    axes[i].set_title('{}'.format(main_directory.split('/')[-1]))

    # Set x and y axis limits for each subplot
    axes[i].set_xlim([0, 1200])
    axes[i].set_ylim([0, 600])
    axes[i].get_xaxis().set_visible(False)
    axes[i].get_yaxis().set_visible(False)
    axes[i].set_aspect('equal', adjustable='box')

    pp1 = plt.Rectangle((480, 300),
                    45, 45, color='red' , alpha = 0.4)
    pp2 = plt.Rectangle((720, 180),
                    45, 45, color='red' , alpha = 0.4)
    pp3 = plt.Rectangle((720, 420),
                    45, 45, color='red' , alpha = 0.4)
    pp4 = plt.Rectangle((720, 300),
                    45, 45, color='red' , alpha = 0.4)
    pp5 = plt.Rectangle((480, 180),
                    45, 45, color='red' , alpha = 0.4)
    pp6 = plt.Rectangle((480, 420),
                    45, 45, color='red' , alpha = 0.4)
    if i == 0:
        axes[i].add_patch(pp1)
        axes[i].add_patch(pp2)
        axes[i].add_patch(pp3)
    else:
        axes[i].add_patch(pp4)
        axes[i].add_patch(pp5)
        axes[i].add_patch(pp6)
    

# Create a colorbar legend for the entire figure
cbar = fig.colorbar(heatmap[-1], ax=axes.ravel().tolist(), shrink=0.6, aspect=10, fraction=0.046, pad=0.02)
cbar.set_label('State Visitation Frequency')

plt.show()

