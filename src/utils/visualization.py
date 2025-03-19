import scipy
import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rcParamsDefault
from typing import List, Union
import pandas as pd 
import numpy as np
from matplotlib.colors import Normalize, ListedColormap,LinearSegmentedColormap


def plot_heatmap(weight_matrix,axis=None,title=None, xlabel=None,ylabel="Locations",xticks = None,xticklabels=None,yticks=None,yticklabels=None,vmin=0,vmax=0.4):
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


    g = sns.heatmap(weight_matrix, cmap=mcolors.LinearSegmentedColormap.from_list("colormap", COLORS, N=100),ax=axis,vmin=vmin, vmax=vmax)
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





def standard_scatters_plot(x,y,x_label = 'Predicted expression', y_label = 'Measured Expression',axes = None):
    pcc = scipy.stats.pearsonr(x,y)
    print(f'The Pearson\'s r for the data is', format(pcc[0], '0.3f'))


    fig=plt.figure(figsize=(4,4) , dpi= 300, facecolor='w', edgecolor='k')
    fig.tight_layout(pad = 1)
    rc = {
                'figure.autolayout' : True,
                'axes.titlesize' : 8 ,
                'axes.titleweight' :'bold',
                
                'figure.titleweight' : 'bold' ,
                'figure.titlesize' : 8 ,
                
                'axes.labelsize' : 8 ,
                'axes.labelpad' : 2 ,
                'axes.labelweight' : 'bold' , 
                'axes.spines.top' : False,
                'axes.spines.right' : False,                
                'xtick.labelsize' : 7 ,
                'ytick.labelsize' : 7 ,
                
                'legend.fontsize' : 7 ,
                'figure.figsize' : (3.5, 3.5/1.6 ) ,          
                
                'xtick.direction' : 'out' ,
                'ytick.direction' : 'out' ,
                
                'xtick.major.size' : 2 ,
                'ytick.major.size' : 2 ,
                
                'xtick.major.pad' : 2,
                'ytick.major.pad' : 2,
                'pdf.fonttype': 42,
                }


    rcParams.update(rcParamsDefault)
    rcParams.update(rc)
    with sns.plotting_context(context='paper',rc=rcParams):
        with sns.axes_style('ticks'):
            sns.regplot(x=x ,y=y ,
                        scatter_kws= {'s':1,'linewidth':0, 'rasterized':True} ,
                        line_kws= {'linewidth':2} ,
                        color= '#0868ac', robust = 1 )

            if not axes:
                axes = plt.gca()

            axes.set_xlabel(x_label)
            axes.set_ylabel(y_label)
            axes.set_title(f"PCC = {pcc[0] : 0.3f} | P < {np.nextafter(0, 1) : 0.0E} | N = {len(x)}"  )
            plt.setp(axes.artists, edgecolor = 'k')
            plt.setp(axes.lines, color='k')
            sns.despine()
            #plt.setp(axes.lines, linewidth=1.5)
            plt.show()


def standard_box_plot(x: Union[str,List] ,
                      y: Union[str,List] ,
                      data: pd.DataFrame ,
                      x_label:str = None,
                      y_label: str= None, 
                      hue=None,
                      axes = None,
                      xlim = None,
                      ylim = None,
                      legend_loc = None,
                      legend_ncol = 1,
                      legend_title = None,
                      *args,
                      **kwargs,
                      ):

    fig=plt.figure(figsize=(4,4) , dpi= 300, facecolor='w', edgecolor='k')
    fig.tight_layout(pad = 1)
    rc = {
                'figure.autolayout' : True,
                'axes.titlesize' : 8 ,
                'axes.titleweight' :'bold',
                
                'figure.titleweight' : 'bold' ,
                'figure.titlesize' : 8 ,
                
                'axes.labelsize' : 8 ,
                'axes.labelpad' : 2 ,
                'axes.labelweight' : 'bold' , 
                'axes.spines.top' : False,
                'axes.spines.right' : False,                
                'xtick.labelsize' : 7 ,
                'ytick.labelsize' : 7 ,
                
                'legend.fontsize' : 7 ,
                'figure.figsize' : (3.5, 3.5/1.6 ) ,          
                
                'xtick.direction' : 'out' ,
                'ytick.direction' : 'out' ,
                
                'xtick.major.size' : 2 ,
                'ytick.major.size' : 2 ,
                
                'xtick.major.pad' : 2,
                'ytick.major.pad' : 2,
                'pdf.fonttype': 42,
                }


    rcParams.update(rcParamsDefault)
    rcParams.update(rc)
    with sns.plotting_context(context='paper',rc=rcParams):
        with sns.axes_style('ticks'):
            sns.boxplot(x=x,y=y,data=data,hue=hue,fliersize=0,whis=0.5,linewidth=1,*args,**kwargs)

            if not axes:
                axes = plt.gca()
            if x_label:
                axes.set_xlabel(x_label)
            if y_label:
                axes.set_ylabel(y_label)

            axes.set_xticklabels(axes.get_xticklabels(),rotation = 0)
            plt.setp(axes.artists, edgecolor = 'k')
            # plt.legend(loc = 'best')
            plt.legend(ncol=2)
            if legend_loc:
                plt.legend(loc=legend_loc, ncol=legend_ncol, title=legend_title)
            else:
                plt.legend(loc='best', ncol=legend_ncol)
            if xlim:
                plt.xlim(xlim)
            if ylim:
                plt.ylim(ylim)
            sns.despine()
            #plt.setp(axes.lines, linewidth=1.5)
            plt.show()
