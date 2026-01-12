from src.data_preprocessing.histone_marks.constants import BASE_MAP
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns 
import matplotlib.colors as mcolors



def convert_one_hot_to_base(v):
    raise NotImplementedError



def plot_tracks(tracks, interval, height=1.5):
  fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
  for ax, (title, y) in zip(axes, tracks.items()):
    ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
    ax.set_title(title)
    sns.despine(top=True, right=True, bottom=True)
  ax.set_xlabel(str(interval))
  plt.tight_layout()



def plot_heatmap(weight_matrix,axis=None,title=None, xlabel=None,ylabel="Locations",xticks = None,xticklabels=None,yticks=None,yticklabels=None,vmin=0.2,vmax=0.4):
    """
    Plots a clustered heatmap of the weights in a given subnetwork.

    Parameters:
    weight_matrix (DataFrame): A matrix containing the weights of edges in the subnetwork.
    output_path (str): The path where the output plot will be saved.

    Returns:
    None
    """
    
    # Define colors for the heatmap
    COLORS = ['#FFFDDF', "#7FCDBB", "#225EA8"]


    g = sns.heatmap(weight_matrix, 
                    cmap=mcolors.LinearSegmentedColormap.from_list("colormap", COLORS, N=100),
                    ax=axis,
                    vmin=vmin, 
                    vmax=vmax,
                    # cbar_kws={"orientation": "horizontal"}
                )
    if xticks:
        g.set_xticks(xticks)

    if yticks:
        g.set_yticks(yticks)


    # Set the font size and name for x-axis tick labels, and rotate them for better visibility
    # g.set_xticklabels(axis.get_xticklabels(), fontsize=6, rotation=50)

    # Set the font size and name for y-axis tick labels
    # g.set_yticklabels(axis.get_yticklabels(), rotation=360)

    # Customize tick parameters for the heatmap axes
    # g.tick_params(axis='both', which='major', length=0)

    # Adjust the colorbar tick parameters if collections exist
    if g.collections:
        g.collections[0].colorbar.ax.tick_params(labelsize=6, length=1)

    # Set labels for the x-axis and y-axis
    if xlabel:
        axis.set_xlabel(xlabel)
    
    if ylabel:
        axis.set_ylabel(ylabel)

    if xticklabels:
        axis.set_xticklabels(xticklabels)

    if yticklabels:
        axis.set_yticklabels(yticklabels)

    g.set_title(title)