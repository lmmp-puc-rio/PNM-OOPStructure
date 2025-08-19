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
        phase_model['pore.pressure'] = (np.random.uniform(0, 1, size=phase_model.Np)) * min(self.network.network['throat.diameter'])
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
            P_diff = abs(pressure[P12[:, 0]] - pressure[P12[:, 1]])/length

            # Reference pressures for region boundaries
            P_ref_0 = 2 * mu_0 * gamma_dot_0 / r_eff
            P_ref_inf = 2 * mu_inf * gamma_dot_inf / r_eff

            Q = np.zeros_like(P_diff)

            # Masks for each region
            mask_A = P_diff <= P_ref_0
            mask_B = (P_diff > P_ref_0) & (P_diff <= P_ref_inf)
            mask_C = P_diff > P_ref_inf

            # Region A: Newtonian (low shear)
            if np.any(mask_A):
                A = pow(r_eff, 4) * P_diff * pi / (8 * mu_0)
                Q[mask_A] = A[mask_A]
            # Region B: Power-law transition
            if np.any(mask_B):
                B1_num = 2 * (1 - n) * pi * pow(mu_0, 3) * pow(gamma_dot_0, 4)
                B1_den = (3 * n + 1) * pow(P_diff, 3)
                B2_num = n * pi * pow(r_eff, 3) * pow(r_eff * P_diff * pow(gamma_dot_0, n - 1), 1 / n)
                B2_den = (3 * n + 1) * pow(2 * mu_0, 1 / n)
                B = (B1_num / B1_den) + (B2_num / B2_den)
                Q[mask_B] = B[mask_B]
            # Region C: Infinite-shear (high shear)
            if np.any(mask_C):
                try:
                    C1_num = 2 * pi * (1 - n) * pow(gamma_dot_0, 4) * (pow(mu_0, 3) - pow(mu_inf, (3 * n + 1) / (n - 1)) * pow(mu_0, (-4) / (n - 1)))
                except Exception:
                    C1_num = 0.0
                C1_den = (3 * n + 1) * pow(P_diff, 3)
                C2_num = pi * pow(r_eff, 3) * P_diff
                C2_den = 3 * mu_inf
                C = (C1_num / C1_den) + (C2_num / C2_den)
                Q[mask_C] = C[mask_C]

            # Replace non-finite values with a large number
            # nanMask = ~np.isfinite(Q)
            # if np.any(nanMask):
            #     Q[nanMask] = 1e-8
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