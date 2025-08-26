r"""
Stokes post-processor for analyzing viscous flow results.

This module provides specialized post-processing capabilities for Stokes flow
simulations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .base_postprocessor import BasePostProcessor
from utils.plots.plotter import Plotter2D, Plotter3D
import openpnm as op


class StokesPostProcessor(BasePostProcessor):
    r"""
    Post-processor for Stokes flow simulation results.
    
    This class provides specialized methods for analyzing and visualizing
    viscous flow results.
    
    Parameters
    ----------
    algorithm_manager : AlgorithmManager
        The algorithm manager containing Stokes flow results
    base_path : str
        Base directory path for saving outputs
    """
    
    def __init__(self, algorithm_manager, base_path):
        super().__init__(algorithm_manager, base_path)
        
    def plot_absolute_permeability(self, algorithm_name, output_file=None):
        r"""
        Plot absolute permeability Pressure vs Flow rate.
        
        Parameters
        ----------
        algorithm_name : str
            Name of the Stokes algorithm
        output_file : str, optional
            Custom output file path
            
        Returns
        -------
        output_file : str
            Path to the saved plot
        """
        alg_dict = self.algorithm_manager.get_algorithm(algorithm_name)
        algorithm = alg_dict['algorithm']
        phase_dict = alg_dict['phase']
        
        results = algorithm.results
        
        plotter = Plotter2D(
            layout='absolute_permeability', 
            title=f'Absolute Permeability {algorithm_name}'
        )
        ax = plotter.ax
        ax.plot(results['pressure'], results['flow_rate'], color=phase_dict['color'])
        
        plotter.apply_layout()
        output_file = output_file or os.path.join(
            self.graph_path, f'absPerm_{algorithm_name}.png'
        )
        plotter.save(output_file)
        return output_file
        
    def plot_pressure_field(self, algorithm_name, output_file=None):
        r"""
        Plot pressure field distribution in the network.
        
        Parameters
        ----------
        algorithm_name : str
            Name of the Stokes algorithm
        output_file : str, optional
            Custom output file path
            
        Returns
        -------
        output_file : str
            Path to the saved plot
        """
        alg_dict = self.algorithm_manager.get_algorithm(algorithm_name)
        algorithm = alg_dict['algorithm']
        phase_dict = alg_dict['phase']
        
        pn = self.algorithm_manager.network.network
        dim = self.algorithm_manager.network.dim
        
        if dim == '3D':
            plotter = Plotter3D(layout='pore_network_3d', title=f'Pressure Field {algorithm_name}')
        else:
            plotter = Plotter2D(layout='pore_network_2d', title=f'Pressure Field {algorithm_name}')
        ax = plotter.ax

        if hasattr(algorithm.algorithm, 'soln') and algorithm.algorithm.soln is not None:
            pressure_field = algorithm.algorithm.soln
            if 'pore.pressure' in pressure_field.keys():
                pressure_field = pressure_field['pore.pressure']
        else:
            phase_model = phase_dict['model']
            if 'pore.pressure' in phase_model.keys():
                pressure_field = phase_model['pore.pressure']
            else:
                raise ValueError(f"No pressure solution found for algorithm {algorithm_name}")
        
        # Convert to MPa
        vmin = np.min(pressure_field) / 1e6
        vmax = np.max(pressure_field) / 1e6
        sc = op.visualization.plot_coordinates(
            pn, color_by=pressure_field/1e6, ax=ax, cmap='jet', vmin=vmin, vmax=vmax
        )

        if hasattr(sc, 'set_cmap'):
            sc.set_cmap('jet')
        if hasattr(sc, 'set_clim'):
            sc.set_clim(vmin, vmax)
        
        cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', shrink=0.8, pad=0)
        cbar.set_label('Pressure [MPa]')

        plotter.apply_layout()
        output_file = output_file or os.path.join(
            self.graph_path, f'Pressure_{algorithm_name}.png'
        )
        plotter.save(output_file)
        return output_file

    def plot_apparent_viscosity(self, algorithm_name, output_file=None):
        r"""
        Plot apparent viscosity vs velocity.
        
        Parameters
        ----------
        algorithm_name : str
            Name of the Stokes algorithm
        output_file : str, optional
            Custom output file path
            
        Returns
        -------
        output_file : str
            Path to the saved plot
        """
        alg_dict = self.algorithm_manager.get_algorithm(algorithm_name)
        algorithm = alg_dict['algorithm']
        phase_dict = alg_dict['phase']
        results = algorithm.results
        plotter = Plotter2D(
            layout='apparent_viscosity', 
            title=f'Apparent Viscosity {algorithm_name}',
            ylabel=r'$\mu_{app} [cP]$'  ,
            ymin = 1,
            ymax = 10
        )
        ax = plotter.ax
        print(results['u'])
        print(results['mu_app'])
        ax.plot(results['u'], results['mu_app']*1000, color=phase_dict['color'])
        ax.set_yscale('log')
        ax.set_xscale('log')
        plotter.apply_layout()
        output_file = output_file or os.path.join(
            self.graph_path, f'appVisc_{algorithm_name}.png'
        )
        plotter.save(output_file)
        return output_file