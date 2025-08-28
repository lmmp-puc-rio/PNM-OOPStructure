r"""
Drainage algorithm implementation for pore network modeling.

This module provides the DrainageAlgorithm class that handles drainage
and imbibition simulations using OpenPNM's Drainage algorithm.
"""

import numpy as np
import openpnm as op
from .base_algorithm import BaseAlgorithm


class DrainageAlgorithm(BaseAlgorithm):
    r"""
    Drainage algorithm for simulating two-phase flow in pore networks.
    
    This class handles drainage and imbibition processes where a non-wetting
    phase displaces a wetting phase through capillary-controlled invasion.
    
    Parameters
    ----------
    network : Network
        The pore network object
    phase : dict
        Phase dictionary containing model and properties
    config : object
        Configuration object with drainage-specific parameters
        
    Attributes
    ----------
    pore_trapped : ndarray
        Boolean array tracking trapped pores from previous simulations
    throat_trapped : ndarray
        Boolean array tracking trapped throats from previous simulations
    """
    
    def __init__(self, network, phase, config):
        super().__init__(network, phase, config)
        self.pore_trapped = np.full(network.network.Np, False)
        self.throat_trapped = np.full(network.network.Nt, False)
        
    def create_algorithm(self):
        r"""
        Create and configure the OpenPNM Drainage algorithm.
        
        Returns
        -------
        algorithm : openpnm.algorithms.Drainage
            Configured drainage algorithm with boundary conditions
        """
        pn = self.network.network
        phase_model = self.phase['model']
        
        inlet_pores = pn.pores('inlet')
        outlet_pores = None
        if self.config.outlet is not None:
            outlet_pores = pn.pores('outlet')

        algorithm = op.algorithms.Drainage(
            network=pn,
            phase=phase_model,
            name=self.config.name
        )
        
        # Set algorithm boundary conditions
        algorithm.set_inlet_BC(pores=inlet_pores, mode='overwrite')
        if outlet_pores is not None:
            algorithm.set_outlet_BC(pores=outlet_pores, mode='overwrite')
            
        self.algorithm = algorithm
        return algorithm
        
    def run(self, pore_trapped=None, throat_trapped=None):
        r"""
        Execute the drainage simulation.
        
        Parameters
        ----------
        pore_trapped : ndarray, optional
            Boolean array of previously trapped pores
        throat_trapped : ndarray, optional
            Boolean array of previously trapped throats
            
        Returns
        -------
        results : dict
            Dictionary containing invasion results and pressure sequence
        """
        if self.algorithm is None:
            self.create_algorithm()
            
        # Set initial trapped states
        if pore_trapped is not None:
            self.pore_trapped = pore_trapped.copy()
        if throat_trapped is not None:
            self.throat_trapped = throat_trapped.copy()
            
        self._apply_initial_conditions()
        pressures = self._calculate_pressure_sequence()
        
        self.algorithm.run(pressures=pressures)
        
        # Update trapped states
        self.pore_trapped = self.algorithm['pore.trapped'].copy()
        self.throat_trapped = self.algorithm['throat.trapped'].copy()
        
        self.results = {
            'pressures': pressures,
            'pore_trapped': self.pore_trapped,
            'throat_trapped': self.throat_trapped
        }
        
        return self.results
        
    def _apply_initial_conditions(self):
        r"""Apply initial invasion conditions based on trapped states."""
        self.algorithm['pore.ic_invaded'] = self.pore_trapped.copy()
        self.algorithm['throat.ic_invaded'] = self.throat_trapped.copy()
        
        if np.any(self.pore_trapped):
            self.algorithm['pore.invaded'] = self.pore_trapped.copy()
        if np.any(self.throat_trapped):
            self.algorithm['throat.invaded'] = self.throat_trapped.copy()
            
    def _calculate_pressure_sequence(self):
        r"""
        Calculate logarithmic pressure sequence for drainage.
        
        Returns
        -------
        pressures : ndarray
            Array of pressures spanning from minimum to maximum entry pressure
        """
        phase_model = self.phase['model']
        p_max = phase_model['throat.entry_pressure'].max()
        p_min = phase_model['throat.entry_pressure'].min()
        samples = self.config.pressures
        
        x = np.linspace(0, 1, samples)
        pressures = p_min * (p_max / p_min) ** x
        
        return pressures
        
    def pc_curve(self, pressures=None):
        r"""
        Generate capillary pressure curve data.
        
        Parameters
        ----------
        pressures : ndarray, optional
            Custom pressure sequence. If None, uses algorithm's pressure sequence
            
        Returns
        -------
        pc_data : object
            Capillary pressure curve data with pc and saturation attributes
        """
        if pressures is None:
            pressures = self._calculate_pressure_sequence()
            
        return self.algorithm.pc_curve(pressures=pressures)
