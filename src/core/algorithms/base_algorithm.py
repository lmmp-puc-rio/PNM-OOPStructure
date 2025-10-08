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