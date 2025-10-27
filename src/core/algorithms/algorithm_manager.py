r"""
Algorithm manager for orchestrating pore network simulations.

This module provides the AlgorithmManager class that creates and coordinates
multiple algorithm instances for complex simulation workflows.
"""

import numpy as np
import openpnm as op
from utils.config_parser import AlgorithmType
from .drainage_algorithm import DrainageAlgorithm
from .stokes_algorithm import StokesAlgorithm
from utils.config_parser import NetworkType, CrossSecType, FluidType

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
        self.config_general = config.network
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
        elif config.type == AlgorithmType.IMBIBITION:
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
                if self.config_general.cross_sec == CrossSecType.TRIANGULAR:
                    self._capillary_pressure_drainage()
                result = algorithm.run(pore_trapped, throat_trapped)
                # Update trapped states for next algorithms
                pore_trapped = result['pore_trapped'].copy()
                throat_trapped = result['throat_trapped'].copy()
            if config.type == AlgorithmType.IMBIBITION:
                if self.config_general.cross_sec == CrossSecType.TRIANGULAR:
                    self._capillary_pressure_imbibition()
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
    
    def _computes_D_S(self, theta_r):
        r"""
        Computes middle variables for cappilary pressure and effective area
        """
        pn = self.network.network
        G = pn["throat.shape_factor"]

        circular_throats = G > 0.079

        # AM -- arc menisci

        theta_r_3 = np.tile(theta_r[:,np.newaxis], (1, 3))

        bi = pn["throat.corner_angles"]


        contains_AM = bi < np.pi/2-theta_r_3
        
        S1 = np.sum( (( np.cos(theta_r_3)*np.cos(theta_r_3+bi) )/(np.sin(bi)) + theta_r_3 + bi - np.pi/2) * contains_AM
                    , axis=1, keepdims=False)
        S2 = np.sum( (( np.cos(theta_r_3+bi) )/(np.sin(bi))) * contains_AM
                    , axis=1, keepdims=False)
        S3 = np.sum( (( np.pi/2 - theta_r_3 - bi )) * contains_AM
                    , axis=1, keepdims=False)
        
        D = S1 - 2*S2*np.cos(theta_r) + S3

        return D, S1
    
    def _compute_capillary_pressure(self, theta_r, D):
        r"""
        Capillary pressure computation
        """
        pn = self.network.network
        r = pn["throat.diameter"]/2
        G = pn["throat.shape_factor"]

        circular_throats = G > 0.079
        Fd = 1 + np.sqrt(1 + ((4*G*D)/(np.cos(theta_r)**2)) )
        Fd = Fd/ (1 + 2*np.sqrt(np.pi*G))

        sigma = self.phases.get_wetting_phase()["model"]["pore.surface_tension"]
        sigma = np.ones_like(theta_r)* sigma[0]
        Pc = sigma * np.cos(theta_r) * (1 + 2*np.sqrt(np.pi*G))
        Pc = Pc/r*Fd

        Pc[circular_throats] = 2 * sigma[circular_throats] * np.cos(theta_r[circular_throats]) /r[circular_throats]

        return Pc
    
    def _capillary_pressure_drainage(self):
        r"""
        Compute the capillary pressure during the drainage process.
        This considers the amount of water that remains as a film on the throat walls, modeled as water remaining at the triangle corners
        https://doi.org/10.1029/2003WR002627   Valvatne and Blunt 2004
        """
        # theta_r -- receding_contact_angle 
        # AM -- arc menisci
        theta_r = self.phases.get_wetting_phase()["model"]["throat.receding_contact_angle"]
        theta_r = np.radians(theta_r)

        D, _ = self._computes_D_S(theta_r)

        Pc = self._compute_capillary_pressure(theta_r, D)

        self.phases.get_non_wetting_phase()["model"].add_model(
                    propname='throat.entry_pressure',
                    model=op.models.misc.constant,
                    value=Pc, 
                    regen_mode='normal'
                )
        

    def _capillary_pressure_imbibition(self):
        r"""
        Compute the capillary pressure during the imbibition process.
        This considers the amount of water that remains as a film on the throat walls, modeled as water remaining at the triangle corners
        https://doi.org/10.1029/2003WR002627   Valvatne and Blunt 2004
        """
        # theta_a -- advancing_contact_angle 
        # AM -- arc menisci
        theta_a = self.phases.get_wetting_phase()["model"]["throat.advancing_contact_angle"]
        theta_a = np.radians(theta_a)
        theta_a = theta_a

        D, _ = self._computes_D_S(theta_a)

        Pc = self._compute_capillary_pressure(theta_a, D)

        self.phases.get_wetting_phase()["model"].add_model(
                    propname='throat.entry_pressure',
                    model=op.models.misc.constant,
                    value=Pc, 
                    regen_mode='normal'
                )
