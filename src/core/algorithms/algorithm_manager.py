r"""
Algorithm manager for orchestrating pore network simulations.

This module provides the AlgorithmManager class that creates and coordinates
multiple algorithm instances for complex simulation workflows.
"""

import numpy as np
from utils.config_parser import AlgorithmType
from .drainage_algorithm import DrainageAlgorithm
from .stokes_algorithm import StokesAlgorithm


class AlgorithmManager:
    r"""
    Manager class for coordinating multiple pore network algorithms.
    
    This class creates and orchestrates the execution of multiple algorithm
    instances, handling dependencies between algorithms (e.g., trapped phase
    states from drainage affecting subsequent simulations).
    
    Parameters
    ----------
    network : Network
        The pore network object
    phases : Phases
        Phases object containing all phase models
    config : object
        Configuration object containing algorithm specifications
        
    Attributes
    ----------
    network : Network
        Reference to the pore network
    phases : Phases
        Reference to the phases object
    algorithms : list
        List of algorithm dictionaries containing instances and metadata
    """
    
    def __init__(self, network, phases, config):
        self.network = network
        self.phases = phases
        self.config = config.algorithm
        self.algorithms = []
        self._create_algorithms()
        
    def _create_algorithms(self):
        r"""Create all algorithm instances based on configuration."""
        for algorithm_config in self.config:
            phase_dict = self._get_phase_dict(algorithm_config.phase)
            algorithm_instance = self._create_algorithm_instance(algorithm_config, phase_dict)
            
            alg_dict = {
                'name': algorithm_config.name,
                'algorithm': algorithm_instance,
                'phase': phase_dict,
                'config': algorithm_config,
            }
                
            self.algorithms.append(alg_dict)
            
    def _get_phase_dict(self, phase_name):
        r"""
        Retrieve phase dictionary by name.
        
        Parameters
        ----------
        phase_name : str
            Name of the phase to retrieve
            
        Returns
        -------
        phase_dict : dict
            Dictionary containing phase information and models
        """
        return next(
            phase for phase in self.phases.phases 
            if phase["name"] == phase_name
        )
        
    def _create_algorithm_instance(self, config, phase_dict):
        r"""
        Create algorithm wrapper instance based on type.
        
        The manager only creates the wrapper instances and lets each
        algorithm class handle its own OpenPNM algorithm creation.
        
        Parameters
        ----------
        config : object
            Algorithm configuration object
        phase_dict : dict
            Phase dictionary with models and properties
            
        Returns
        -------
        algorithm : BaseAlgorithm
            Algorithm wrapper instance
        """
        if config.type == AlgorithmType.DRAINAGE:
            return DrainageAlgorithm(self.network, phase_dict, config)
        elif config.type == AlgorithmType.STOKES:
            return StokesAlgorithm(
                self.network, phase_dict, config, self.phases,
                domain_length=None, domain_area=None
            )
        else:
            raise ValueError(f"Unsupported algorithm type: {config.type}")
        
    def run_all(self):
        r"""
        Execute all algorithms in sequence.
        
        This method runs algorithms in the order they were configured,
        passing trapped state information from drainage algorithms to
        subsequent algorithms as needed.
        
        Returns
        -------
        results : list
            List of result dictionaries from each algorithm
        """
        pore_trapped = np.full(self.network.network.Np, False)
        throat_trapped = np.full(self.network.network.Nt, False)
        results = []
        
        for alg_dict in self.algorithms:
            algorithm = alg_dict['algorithm']
            config = alg_dict['config']
            
            if config.type == AlgorithmType.DRAINAGE:
                result = algorithm.run(pore_trapped, throat_trapped)
                # Update trapped states for next algorithms
                pore_trapped = result['pore_trapped'].copy()
                throat_trapped = result['throat_trapped'].copy()
            elif config.type == AlgorithmType.STOKES:
                result = algorithm.run()
                
            results.append(result)
            
        return results
        
    def get_algorithm(self, name):
        r"""
        Retrieve algorithm instance by name.
        
        Parameters
        ----------
        name : str
            Name of the algorithm to retrieve
            
        Returns
        -------
        alg_dict : dict
            Algorithm dictionary containing instance and metadata
        """
        return next(
            alg for alg in self.algorithms 
            if alg['name'] == name
        )
