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
            
    def calculate_permeability(self):
        r"""
        Calculate intrinsic permeability using unit viscosity reference flow.
        
        The permeability is calculated by running a reference Stokes flow
        simulation with unit viscosity and unit pressure drop.
        
        Returns
        -------
        K : float
            Intrinsic permeability in m²
        """
        pn = self.network.network
        R = pn['throat.diameter']/2
        L = pn['throat.length']
        reference_phase = op.phase.Phase(network=pn)
        reference_phase.add_model_collection(op.models.collections.physics.basic)
        reference_phase['pore.viscosity'] = 1.0
        reference_phase['throat.hydraulic_conductance'] = np.pi*R**4/(8*L)
        
        inlet_pores = pn.pores('inlet')
        outlet_pores = pn.pores('outlet')

        flow = op.algorithms.StokesFlow(network=pn, phase=reference_phase)
        flow.set_value_BC(pores=inlet_pores, values=1)
        flow.set_value_BC(pores=outlet_pores, values=0)
        flow.run()
        
        # Calculate permeability: K = Q * L * μ / (A * ΔP)
        # With μ = 1 and ΔP = 1, this simplifies to K = Q * L / A
        Q = flow.rate(pores=inlet_pores, mode='group')[0]
        K = Q * self.domain_length / self.domain_area
        
        return K

    def calculate_porosity(self): 
        pn = self.network.network
        Vol_void = np.sum(pn['pore.volume'])+np.sum(pn['throat.volume'])
        inlet_pores = pn.pores('inlet')
        outlet_pores = pn.pores('outlet')
        A = op.topotools.get_domain_area(pn, inlets=inlet_pores, outlets=outlet_pores)
        L = op.topotools.get_domain_length(pn, inlets=inlet_pores, outlets=outlet_pores)
        Vol_bulk = A * L
        Poro = Vol_void / Vol_bulk
        print(f'The value of Porosity is: {Poro:.2f}')
        
        return Poro