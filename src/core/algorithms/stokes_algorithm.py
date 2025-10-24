r"""
Stokes flow algorithm implementation for pore network modeling.

This module provides the StokesAlgorithm class that handles steady-state
viscous flow simulations with support for non-Newtonian fluid behavior.
"""

import numpy as np
import openpnm as op
from .base_algorithm import BaseAlgorithm
from utils.config_parser import NetworkType, CrossSecType, FluidType

class StokesAlgorithm(BaseAlgorithm):
    r"""
    Stokes flow algorithm for simulating viscous flow in pore networks.
    
    This class handles steady-state flow simulations with support for
    non-Newtonian fluid behavior and apparent viscosity calculations.
    
    Parameters
    ----------
    network : Network
        The pore network object
    phase : dict
        Phase dictionary containing model and properties
    config : object
        Configuration object with Stokes flow parameters
    phases : Phases
        Phases object for accessing conductance models
        
    Attributes
    ----------
    phases : Phases
        Reference to phases object for model management
    domain_length : float
        Domain length for permeability calculations
    domain_area : float
        Domain cross-sectional area for permeability calculations
    """
    
    def __init__(self, network, phase, config, phases, domain_length=None, domain_area=None):
        super().__init__(network, phase, config)
        self.config_general = config.network
        self.phases = phases
        self.domain_length = domain_length
        self.domain_area = domain_area
        
    def create_algorithm(self):
        r"""
        Create and configure the OpenPNM StokesFlow algorithm.
        
        Returns
        -------
        algorithm : openpnm.algorithms.StokesFlow
            Configured Stokes flow algorithm
        """
        pn = self.network.network
        phase_model = self.phase['model']
        
        # Calculate domain properties for 3D networks if not already set
        if self.network.dim == '3D' and self.domain_length is None:
            self.domain_length = op.topotools.get_domain_length(
                pn, inlets=pn['pore.inlet'], outlets=pn['pore.outlet']
            )
            self.domain_area = op.topotools.get_domain_area(
                pn, inlets=pn['pore.inlet'], outlets=pn['pore.outlet']
            )
        
        algorithm = op.algorithms.StokesFlow(
            network=pn,
            phase=phase_model,
            name=self.config.name
        )
        
        self.algorithm = algorithm
        return algorithm
        
    def run(self):
        r"""
        Execute the Stokes flow simulation with pressure sweep.
        
        This method performs a series of flow simulations across a range
        of inlet pressures to characterize non-Newtonian flow behavior.
        
        Returns
        -------
        results : dict
            Dictionary containing flow rates, apparent viscosities, and shear rates
        """
        if self.algorithm is None:
            self.create_algorithm()
        self.algorithm.settings["f_rtol"] = 1e-6
        self.algorithm.settings["x_rtol"] = 1e-6
        if self.config_general.cross_sec == FluidType.NONNEWT:
            self._setup_non_newtonian_conductance()
        
        pn = self.network.network
        phase_model = self.phase['model']
        
        K = self.network.calculate_permeability()
        D = np.mean(pn['throat.diameter'])
        
        p_sequence = np.linspace(
            self.config.initial_pressure,
            self.config.final_pressure,
            self.config.pressures
        )
        
        inlet_pores = pn.pores('inlet')
        outlet_pores = pn.pores('outlet')
        
        Q = np.array([])
        mu_app = np.array([])
        
        for p in p_sequence:
            self.algorithm.set_value_BC(pores=inlet_pores, values=p, mode='overwrite')
            self.algorithm.set_value_BC(pores=outlet_pores, values=0, mode='overwrite')
            self.algorithm.run(x0=phase_model['pore.pressure'])
            
            phase_model.update(self.algorithm.soln)
            flow_rate = self.algorithm.rate(pores=inlet_pores, mode='group')[0]
            Q = np.append(Q, flow_rate)
            
        mu_app = K * self.domain_area * p_sequence / (Q * self.domain_length)
        u = Q / self.domain_area
        
        self.results = {
            'pressure': p_sequence,
            'dP/dx': p_sequence / self.domain_length,
            'flow_rate': Q,
            'mu_app': mu_app,
            'u': u
        }
        
        return self.results
        
    def _setup_non_newtonian_conductance(self):
        r"""Configure non-Newtonian conductance model for the phase."""
        phase_model = self.phase['model']
        conductance = 'throat.non_newtonian_conductance'
        
        self.phases.add_non_newtonian_conductance_model(phase_model)
        phase_model.regenerate_models()
        self.algorithm.settings._update({'conductance': conductance})
