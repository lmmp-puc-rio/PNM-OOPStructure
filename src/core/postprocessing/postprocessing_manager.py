r"""
Post-processing manager for coordinating analysis workflows.

This module provides the PostProcessingManager class that creates and coordinates
multiple post-processing instances for comprehensive result analysis.
"""

from utils.config_parser import AlgorithmType
from .base_postprocessor import BasePostProcessor
from .drainage_postprocessor import DrainagePostProcessor
from .stokes_postprocessor import StokesPostProcessor


class PostProcessingManager:
    r"""
    Manager class for coordinating multiple post-processing workflows.
    
    This class creates and orchestrates the execution of multiple post-processing
    instances, providing a unified interface for analyzing different algorithm results.
    
    Parameters
    ----------
    algorithm_manager : AlgorithmManager
        The algorithm manager containing all simulation results
    base_path : str
        Base directory path for saving outputs
        
    Attributes
    ----------
    algorithm_manager : AlgorithmManager
        Reference to the algorithm manager
    base_path : str
        Base directory for outputs
    base_processor : BasePostProcessor
        General post-processor for network visualization
    drainage_processor : DrainagePostProcessor or None
        Drainage-specific post-processor if drainage algorithms exist
    stokes_processor : StokesPostProcessor or None
        Stokes-specific post-processor if Stokes algorithms exist
    """
    
    def __init__(self, algorithm_manager, base_path):
        self.algorithm_manager = algorithm_manager
        self.base_path = base_path
        
        self.base_processor = BasePostProcessor(algorithm_manager, base_path)
        self.drainage_processor = None
        self.stokes_processor = None
        
        self._initialize_processors()
        
    def _initialize_processors(self):
        r"""Initialize specialized processors based on algorithm types."""
        has_drainage = False
        has_stokes = False

        for alg_dict in self.algorithm_manager.algorithms:
            config = alg_dict['config']
            if config.type == AlgorithmType.DRAINAGE:
                has_drainage = True
            elif config.type == AlgorithmType.STOKES:
                has_stokes = True
                
        if has_drainage:
            self.drainage_processor = DrainagePostProcessor(
                self.algorithm_manager, self.base_path
            )
            
        if has_stokes:
            self.stokes_processor = StokesPostProcessor(
                self.algorithm_manager, self.base_path
            )
            
    def plot_network(self, **kwargs):
        r"""
        Plot the pore network topology.
        
        Delegates to the base processor for general network visualization.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to BasePostProcessor.plot_network()
            
        Returns
        -------
        output_file : str
            Path to the saved network plot
        """
        return self.base_processor.plot_network(**kwargs)
        
    def plot_network_tutorial(self, **kwargs):
        r"""
        Plot the pore network in tutorial style with labels.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to the network tutorial plotting method
            
        Returns
        -------
        output_file : str
            Path to the saved tutorial network plot
        """
        return self.base_processor.plot_network_tutorial(**kwargs) 