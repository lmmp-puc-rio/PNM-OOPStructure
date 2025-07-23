from utils.config_parser import PhaseModel
import openpnm as op
import numpy as np
class Phases:
    r"""
    Phases(network, config)

    Creates and manages phase objects for OpenPNM simulations.

    Parameters
    ----------
    network : Network
        The network object to which phases are attached.
    config : ProjectConfig
        Parsed configuration object from ConfigParser.

    Attributes
    ----------
    phases : list of dict
        List of phase dictionaries with 'name', 'model', and 'color'.
    """
    def __init__(self, network, config):
        self.config = config.phases
        self.network = network
        self.phases = []
        for dict_phase in self.config:
            self.phases.append(
                dict(
                    name=dict_phase.name,
                    model=self._create_phase_model(dict_phase),
                    color=dict_phase.color,
                    config=dict_phase
                )
            )

    def _create_phase_model(self, raw: dict):
        r"""
        Creates a phase model (e.g., Water, Air) and attaches physics models.

        Parameters
        ----------
        raw : dict
            Phase configuration dictionary.

        Returns
        -------
        phase_model : openpnm.phase.Phase
            The created phase model object.
        """
        if raw.model == PhaseModel.WATER:
            phase_model = op.phase.Water(network=self.network.network, name=raw.name)
            phase_model.add_model_collection(op.models.collections.phase.water)
        elif raw.model == PhaseModel.AIR:
            phase_model = op.phase.Air(network=self.network.network, name=raw.name)
            phase_model.add_model_collection(op.models.collections.phase.air)
        else:
            raise ValueError(f"PhaseModel: {raw.model}")
        phase_model.add_model_collection(op.models.collections.physics.basic)
        for prop in raw.properties.keys():
            if prop.split('.')[0] in ['throat', 'pore']:
                phase_model.add_model(
                    propname=prop,
                    model=op.models.misc.constant,
                    value=raw.properties[prop]
                )
            elif prop.split('.')[0] in ['param']:
                phase_model[prop] = raw.properties[prop]
            else:
                raise ValueError(f"Unknown property prefix in {prop}")
        phase_model.regenerate_models()
        return phase_model

    def get_model(self, name):
        r"""
        Returns the phase model by name.

        Parameters
        ----------
        name : str
            Name of the phase.

        Returns
        -------
        model : openpnm.phase.Phase
            The phase model object.
        """
        return next(phase["model"] for phase in self.phases if phase["name"] == name)

    def get_wetting_phase(self):
        r"""
        Returns the dictionary of the wetting phase (contact angle < 90).

        Returns
        -------
        phase : dict
            Dictionary containing 'name', 'model', and 'color' for the wetting phase.
        """
        for phase in self.phases:
            model = phase['model']
            if model['pore.contact_angle'][0] < 90:
                return phase
        return None

    def get_non_wetting_phase(self):
        r"""
        Returns the dictionary of the non-wetting phase (contact angle >= 90).

        Returns
        -------
        phase : dict
            Dictionary containing 'name', 'model', and 'color' for the non-wetting phase.
        """
        for phase in self.phases:
            model = phase['model']
            if model['pore.contact_angle'][0] >= 90:
                return phase
        return None

    def add_conduit_conductance_model(self, phase_model):
        r"""
        Adds the conduit conductance model to the given phase model.

        Parameters
        ----------
        phase_model : openpnm.phase.Phase
            The phase model to which the conduit conductance model is added.
        """
        model_mp_cond = op.models.physics.multiphase.conduit_conductance
        phase_model.add_model(
            model=model_mp_cond,
            propname='throat.conduit_hydraulic_conductance',
            throat_conductance='throat.hydraulic_conductance',
            mode='medium',
            regen_mode='deferred'
        )
    
    def add_non_newtonian_conductance_model(self, phase_model):
        r"""
        Adds a non-Newtonian conductance model to the given phase model.

        Parameters
        ----------
        phase_model : openpnm.phase.Phase
            The phase model to which the non-Newtonian conductance model is added.
        """
        def _non_newtonian_conductance(prop, pressure):
            r"""
            Calculates throat conductance for non-Newtonian fluids using a piecewise model.
            
            Parameters
            ----------
            prop : ndarray
                Non-Newtonian volumetric flow rate.
            pressure : ndarray
                Array of pore pressures.

            Returns
            -------
            g : ndarray
                Calculated throat conductance values.
                
            Reference
            ---------
            https://doi.org/10.1016/j.compgeo.2025.107142
            """
            Q = prop
            pressure = phase_model[pressure]

            # Get throat connections and pressure differences
            P12 = self.network.network['throat.conns']
            P_diff = abs(pressure[P12[:, 0]] - pressure[P12[:, 1]])

            g = Q / P_diff

            # Replace non-finite values with a large number
            nanMask = ~np.isfinite(g)
            if np.any(nanMask):
                g[nanMask] = 1e16
            return g
        
        self.add_non_newtonian_volumetric_flow_rate_model(phase_model)
        phase_model.add_model(
            propname='throat.non_newtonian_conductance',
            model=op.models.misc.generic_function,
            func=_non_newtonian_conductance,
            prop="throat.non_newtonian_volumetric_flow_rate",
            pressure='pore.pressure',
            regen_mode='deferred'
        )
        
    def add_non_newtonian_volumetric_flow_rate_model(self, phase_model):
        r"""
        Adds a non-Newtonian volumetric flow rate model to the given phase model.

        Parameters
        ----------
        phase_model : openpnm.phase.Phase
            The phase model to which the non-Newtonian volumetric flow rate model is added.
        """
        def _non_newtonian_volumetric_flow_rate(prop, mu_0, power_law_index,
                                      mu_inf, gamma_dot_inf, gamma_dot_0, diameter, length):
            r"""
            Calculates throat volumetric flow rate for non-Newtonian fluids using a piecewise model.

            Parameters
            ----------
            prop : ndarray
                Array of pore pressures.
            mu_0 : float
                Zero-shear viscosity.
            power_law_index : float
                Power law index (n).
            mu_inf : float
                Infinite-shear viscosity.
            gamma_dot_inf : float
                Infinite-shear rate.
            gamma_dot_0 : float
                Zero-shear rate.
            diameter : ndarray
                Throat diameters.

            Returns
            -------
            Q : ndarray
                Calculated throat volumetric flow rate values.

            Reference
            ---------
            https://doi.org/10.1016/j.compgeo.2025.107142
            """
            pressure = prop
            mu_0 = phase_model[mu_0]
            n = phase_model[power_law_index]
            mu_inf = phase_model[mu_inf]
            gamma_dot_inf = phase_model[gamma_dot_inf]
            gamma_dot_0 = phase_model[gamma_dot_0]
            diameter = phase_model[diameter]
            length = phase_model[length]
            r_eff = diameter / 2
            pi = np.pi

            # Get throat connections and pressure differences
            P12 = self.network.network['throat.conns']
            
            P_throat = (pressure[P12[:, 0]] + pressure[P12[:, 1]])/2
            P_critico = 5.0e+7
            # Reference pressures for region boundaries
            P_diff = abs(pressure[P12[:, 0]] - pressure[P12[:, 1]])/length

            Q = np.zeros_like(P_throat)
        
            # Masks for each region
            mask_liq = P_throat <= P_critico
            mask_gas = P_throat > P_critico

            mu_gas = 0.00009
            mu_liq = 0.00011
            # Region A: Newtonian (low shear)
            if np.any(mask_liq):
                A = pow(r_eff, 4) * P_diff * pi / (8 * mu_liq)
                Q[mask_liq] = A[mask_liq]

            if np.any(mask_gas):
                B = pow(r_eff, 4) * P_diff * pi / (8 * mu_gas)
                Q[mask_gas] = B[mask_gas]
            return Q

        phase_model.add_model(
            propname='throat.non_newtonian_volumetric_flow_rate',
            model=op.models.misc.generic_function,
            func=_non_newtonian_volumetric_flow_rate,
            prop="pore.pressure",
            mu_0="param.mu_0",
            power_law_index="param.power_law_index",
            mu_inf="param.mu_inf",
            gamma_dot_inf="param.gamma_dot_inf",
            gamma_dot_0="param.gamma_dot_0",
            diameter='throat.diameter',
            length='throat.length',
            regen_mode='deferred'
        )