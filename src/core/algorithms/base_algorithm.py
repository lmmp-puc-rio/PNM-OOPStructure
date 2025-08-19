r"""
Base algorithm class for pore network modeling.

This module provides the abstract base class that defines the interface
for all algorithm implementations in the pore network modeling framework.
"""

from abc import ABC, abstractmethod
import numpy as np
import openpnm as op


class BaseAlgorithm(ABC):
    r"""
    Abstract base class for all pore network algorithms.
    
    This class defines the common interface that all algorithm implementations
    must follow, ensuring consistency across different physics solvers.
    
    Parameters
    ----------
    network : Network
        The pore network object containing network topology and properties
    phase : dict
        Dictionary containing phase information and models
    config : object
        Configuration object containing algorithm-specific parameters
    
    Attributes
    ----------
    network : Network
        Reference to the pore network
    phase : dict
        Phase dictionary with models and properties
    config : object
        Algorithm configuration parameters
    algorithm : openpnm.Algorithm or None
        The OpenPNM algorithm instance, created by subclasses
    """
    
    def __init__(self, network, phase, config):
        self.network = network
        self.phase = phase
        self.config = config
        self.algorithm = None
        self.results = {}
        
    @abstractmethod
    def create_algorithm(self):
        r"""
        Create the OpenPNM algorithm instance.
        
        This method must be implemented by subclasses to create the specific
        algorithm type (e.g., Drainage, StokesFlow) with appropriate settings.
        
        Returns
        -------
        algorithm : openpnm.Algorithm
            The configured OpenPNM algorithm instance
        """
        pass
        
    @abstractmethod
    def run(self, **kwargs):
        r"""
        Execute the algorithm simulation.
        
        This method must be implemented by subclasses to run the specific
        algorithm with appropriate parameters and boundary conditions.
        
        Parameters
        ----------
        **kwargs
            Algorithm-specific parameters
            
        Returns
        -------
        results : dict
            Dictionary containing simulation results
        """
        pass
        
    def _setup_boundary_conditions(self, inlet_label, outlet_label=None):
        r"""
        Set up inlet and outlet boundary condition labels on the network.
        
        Parameters
        ----------
        inlet_label : str
            Label identifying inlet pores (e.g., 'left', 'top')
        outlet_label : str, optional
            Label identifying outlet pores (e.g., 'right', 'bottom')
        """
        pn = self.network.network
        inlet_pores = pn.pores(inlet_label)
        pn['pore.inlet'] = np.isin(pn.Ps, inlet_pores)
        conns = pn['throat.conns']
        inlet_inlet_throats = pn['pore.inlet'][conns[:, 0]] & pn['pore.inlet'][conns[:, 1]]
        
        if outlet_label is not None:
            outlet_pores = pn.pores(outlet_label)
            pn['pore.outlet'] = np.isin(pn.Ps, outlet_pores)
            
            outlet_outlet_throats = pn['pore.outlet'][conns[:, 0]] & pn['pore.outlet'][conns[:, 1]]
        op.topotools.trim(network=pn, throats=inlet_inlet_throats | outlet_outlet_throats)
        pn.regenerate_models()
            
