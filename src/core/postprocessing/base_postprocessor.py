r"""
Base post-processor class for pore network modeling.

This module provides the abstract base class that defines the common interface
and shared functionality for all post-processing implementations.
"""

from abc import ABC
import os
import openpnm as op
from utils.plots.plotter import Plotter2D, Plotter3D
import matplotlib.pyplot as plt
import numpy as np

class BasePostProcessor(ABC):
    r"""
    Abstract base class for all pore network post-processors.
    
    This class defines common functionality shared across different post-processing
    implementations and ensures consistent interfaces.
    
    Parameters
    ----------
    algorithm_manager : AlgorithmManager
        The algorithm manager containing network, phases, and results
    base_path : str
        Base directory path for saving outputs
        
    Attributes
    ----------
    algorithm_manager : AlgorithmManager
        Reference to the algorithm manager
    base_path : str
        Base directory for outputs
    graph_path : str
        Directory path for saving graphs
    video_path : str
        Directory path for saving videos
    frame_path : str
        Directory path for saving animation frames
    """
    
    def __init__(self, algorithm_manager, base_path):
        self.algorithm_manager = algorithm_manager
        self.base_path = base_path
        
        project_name = algorithm_manager.network.project_name
        results_path = os.path.join(base_path, 'results', project_name)
        self.graph_path = os.path.join(results_path, 'graphs')
        self.video_path = os.path.join(results_path, 'videos')
        self.frame_path = os.path.join(self.video_path, 'frames')
        
        os.makedirs(self.graph_path, exist_ok=True)
        os.makedirs(self.frame_path, exist_ok=True)
        
    def plot_network(self, lwidth=3, msize=100):
        r"""
        Plot the pore network topology.
        
        This method creates a visualization of the pore network structure.
        
        Parameters
        ----------
        lwidth : float, default 3
            Maximum line width for throat visualization
        msize : float, default 100
            Maximum marker size for pore visualization
            
        Returns
        -------
        output_file : str
            Path to the saved network plot
        """
        pn = self.algorithm_manager.network.network
        dim = self.algorithm_manager.network.dim
        
        linewidth = pn['throat.diameter'] / pn['throat.diameter'].max() * lwidth
        markersize = pn['pore.diameter'] / pn['pore.diameter'].max() * msize
        
        if dim == '3D':
            plotter = Plotter3D(layout='pore_network_3d')
        else:
            plotter = Plotter2D(layout='pore_network_2d')
        ax = plotter.ax
        
        op.visualization.plot_coordinates(
            pn, markersize=markersize, c='b', zorder=2, alpha=0.8, ax=ax
        )
        op.visualization.plot_connections(
            pn, linewidth=linewidth, c='b', zorder=1, alpha=0.8, ax=ax
        )
        
        plotter.apply_layout()
        output_file = os.path.join(
            self.graph_path, 
            f'Network_{self.algorithm_manager.network.project_name}.png'
        )
        plotter.save(output_file)
        return output_file
        
    def plot_network_tutorial(self, font_size=1, line_width=0.2, node_size=10, 
                             output_file=None, dpi=600):
        r"""
        Plot the pore network in tutorial style with labels.
        
        This method creates a network visualization similar to OpenPNM's
        tutorial plots, showing the network structure with labels.
        
        Parameters
        ----------
        font_size : float, default 1
            Font size for labels and annotations
        line_width : float, default 0.2
            Width of throat connections
        node_size : float, default 10
            Size of pore markers
        output_file : str, optional
            Custom output file path. If None, uses default naming
        dpi : int, default 600
            Resolution for saved image
            
        Returns
        -------
        output_file : str
            Path to the saved tutorial network plot
        """
        
        pn = self.algorithm_manager.network.network
        
        fig, ax = plt.subplots()
        
        throat_labels = np.empty_like(pn.Ts, dtype=str)
        throat_labels[:] = ''
        
        op.visualization.plot_tutorial(
            pn, 
            font_size=font_size,
            line_width=line_width,
            node_size=node_size,
            throat_labels=throat_labels
        )
        
        fig = plt.gcf()
        output_file = output_file or os.path.join(
            self.graph_path, 
            f'Network_tutorial_{self.algorithm_manager.network.project_name}.png'
        )
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        return output_file
        
    def _plot_pores_and_throats(self, pn, pores=None, throats=None, markersize=None, 
                                linewidth=None, ax=None, **kwargs):
        r"""
        Helper method to plot pores and throats with flexible styling.
        
        Parameters
        ----------
        pn : openpnm.Network
            The pore network
        pores : array_like, optional
            Pore indices to plot
        throats : array_like, optional
            Throat indices to plot
        markersize : array_like, optional
            Marker sizes for pores
        linewidth : array_like, optional
            Line widths for throats
        ax : matplotlib.Axes
            Axes object for plotting
        **kwargs
            Additional plotting arguments
        """
        if throats is not None and throats.any():
            op.visualization.plot_connections(
                pn, throats, zorder=1, ax=ax, **kwargs,
                linewidth=linewidth[throats] if linewidth is not None else None,
            )
            
        if pores is not None and pores.any():
            op.visualization.plot_coordinates(
                pn, pores, zorder=2, ax=ax, **kwargs,
                markersize=markersize[pores] if markersize is not None else None,
            )
