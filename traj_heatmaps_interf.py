### Purpose: to create heatmaps of interaction forces.
### Method: Computes and bins the interaction forces for each trajectory in a given directory. Can specify different subdirectories to compare different trajectories collected from different baselines (i.e. bc-lstm-gmm, co-gail, etc.). 

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib import rcParams 
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib.pyplot as plt
from cooperative_transport.gym_table.envs.utils import L

def compute_interaction_forces(table_state, f1, f2):

    table_center_to_player1 = np.array(
            [
                table_state[0] + (L/2) * table_state[2],
                table_state[1] + (L/2) * table_state[3],
            ]
        )
    table_center_to_player2 = np.array(
        [
            table_state[0] - (L/2) * table_state[2],
            table_state[1] - (L/2) * table_state[3],
        ]
    )
    inter_f = (f1 - f2) @ (
            table_center_to_player1 - table_center_to_player2
    )
    return inter_f

# Binning function
def bin_coordinates(coord):
    return (int(coord[:, 0] / 100), int(coord[:, 1] / 100))


# Place subdirectories containing trajectories in a list called 'main_directories'
main_directories = [
    '/home/eleyng/table-carrying-ai/results/hil-dp/Demonstrations',
    '/home/eleyng/table-carrying-ai/results/hil-dp/CoDP-H',
    '/home/eleyng/table-carrying-ai/results/hil-dp/CoDP',
    '/home/eleyng/table-carrying-ai/results/hil-dp/BC-LSTM-GMM',
    '/home/eleyng/table-carrying-ai/results/hil-dp/Co-GAIL',
    '/home/eleyng/table-carrying-ai/results/hil-dp/VRNN'
]


# Create a grid of size (1200, 600)
grid_size = (1200, 600)

# Define bin size
bin_size = 50

# Create bins
x_bins = np.arange(0, grid_size[0] + bin_size, bin_size)
y_bins = np.arange(0, grid_size[1] + bin_size, bin_size)

# Create empty heatmap
heatmap = np.zeros((len(y_bins) - 1, len(x_bins) - 1))

fig, axes = plt.subplots(1, 6, figsize=(15, 3), sharex=True, sharey=True)  # Create subplots with 1 row and 5 columns
plt.tight_layout()  # Adjust subplot spacing

# Loop through the main directories
for i, main_directory in enumerate(main_directories):

    # Loop through subdirectories in the main directory
    for subdir in os.listdir(main_directory):
        subdir_path = os.path.join(main_directory, subdir)

        # Check if the item is a directory
        if os.path.isdir(subdir_path):

            for file_path in glob.glob(os.path.join(subdir_path, '*.npz')):

                # Extract trajectory of states
                data = dict(np.load(file_path, allow_pickle=True))  # Assuming states are stored in a CSV format
                states = data['states']
                action = data['actions']

                if 'fluency' in data.keys():
                    inter_f = data['fluency'].item()['inter_f'] if main_directory.split('/')[-1] not in ['CoDP-H', 'CoDP'] else data['fluency'][0]['inter_f']


                # Iterate over states and find corresponding interaction outputs
                for t in range(states.shape[0]):
                    state_x = states[t, 0]  # Assuming state coordinates are in (x, y) format
                    state_y = states[t, 1]
                    if "fluency" in data.keys():
                        inter_f_t = np.abs(inter_f[t])
                    else:
                        inter_f_t = np.abs(compute_interaction_forces(states[t], action[t, :2], action[t, 2:]))

                    # Find bin indices for the state
                    x_bin_index = np.searchsorted(x_bins, state_x, side='right') - 1
                    y_bin_index = np.searchsorted(y_bins, state_y, side='right') - 1

                    # Assign state to appropriate bin based on interaction output
                    if 0 < inter_f_t < 25:
                        heatmap[y_bin_index, x_bin_index] += 0  # Bin for greater than 30
                    elif 25 < inter_f_t < 50:
                        heatmap[y_bin_index, x_bin_index] += 1  # Bin for greater than 30
                    elif 50 < inter_f_t < 70:
                        heatmap[y_bin_index, x_bin_index] += 2  # Bin for greater than 30
                    elif inter_f_t >= 70:
                        heatmap[y_bin_index, x_bin_index] += 3  # Bin for greater than 70
                    else:
                        heatmap[y_bin_index, x_bin_index] += 0  # Bin for less than 30

    # Plot heatmap
    axes[i].imshow(heatmap, cmap='viridis', extent=[0, grid_size[0], 0, grid_size[1]], origin='lower')
    # axes[i].colorbar(label='Interaction Output')
    # axes[i].xlabel('X')
    # axes[i].ylabel('Y')
    # axes[i].title('Heatmap of Interaction Outputs with States')
    # axes[i].show()

    # Plot a heat map on the corresponding subplot
    # heatmap = axes[i].hist2d(df['x'], df['y'], bins=[100,50], range=[[0,1200],[0, 600]], cmap='viridis', norm=mcolors.PowerNorm(vmin=0.01, vmax=800
    # , gamma=0.45))
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
# cbar = fig.colorbar(heatmap[-1], ax=axes.ravel().tolist(), shrink=0.6, aspect=10, fraction=0.046, pad=0.02)
# cbar.set_label('State Visitation Frequency')

plt.show()

