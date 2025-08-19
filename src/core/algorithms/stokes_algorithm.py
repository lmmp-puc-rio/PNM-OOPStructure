r"""
Stokes flow algorithm implementation for pore network modeling.

This module provides the StokesAlgorithm class that handles steady-state
viscous flow simulations with support for non-Newtonian fluid behavior.
"""

import numpy as np
import openpnm as op
from .base_algorithm import BaseAlgorithm


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
        
        # Use shared boundary condition setup
        self._setup_boundary_conditions(self.config.inlet, self.config.outlet)
        
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
        
        #TODO implement Phase.TYPE to define phase models and properties
        self._setup_non_newtonian_conductance()
        
        pn = self.network.network
        phase_model = self.phase['model']
        
        K = self.calculate_permeability()
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
        gamma_dot = u / D
        
        self.results = {
            'pressure': p_sequence,
            'dP/dx': p_sequence / self.domain_length,
            'flow_rate': Q,
            'mu_app': mu_app,
            'gamma_dot': gamma_dot
        }
        
        return self.results
        
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
    
        reference_phase = op.phase.Phase(network=pn)
        reference_phase['pore.viscosity'] = 1.0
        reference_phase.add_model_collection(op.models.collections.physics.basic)
        reference_phase.regenerate_models()
        
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
        
    def _setup_non_newtonian_conductance(self):
        r"""Configure non-Newtonian conductance model for the phase."""
        phase_model = self.phase['model']
        conductance = 'throat.non_newtonian_conductance'
        
        self.phases.add_non_newtonian_conductance_model(phase_model)
        phase_model.regenerate_models()
        self.algorithm.settings._update({'conductance': conductance})
