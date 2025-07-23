from utils.config_parser import NetworkType
import numpy as np
import openpnm as op

class Network:
    r"""
    Network(config)

    Creates and manages a pore network for OpenPNM simulations.

    Parameters
    ----------
    config : ProjectConfig
        Parsed configuration object from ConfigParser.

    Attributes
    ----------
    network : openpnm.network.Network
        The OpenPNM network object.
    project_name : str
        Name of the project.
    dim : str
        Dimensionality of the network ('2D' or '3D').
    """
    def __init__(self, config):
        self.config = config.network
        self.project_name = getattr(self.config, 'project_name', 'project')
        np.random.seed(self.config.seed)
        self.network = self._create_network()
        self.dim = '2D'
        # Determine dimensionality based on pore coordinates
        if len(np.unique(self.network['pore.coords'].T[2])) > 1:
            self.dim = '3D'

    def _create_network(self):
        r"""
        Creates the network based on the configuration type.
        Returns
        -------
        network : openpnm.network.Network
            The generated OpenPNM network object.
        """
        if self.config.type == NetworkType.CUBIC:
            return self._create_cubic()
        elif self.config.type == NetworkType.IMPORTED:
            return self._create_imported()
        else:
            raise ValueError(f"NetworkType: {self.config.type}")

    def _create_cubic(self):
        r"""
        Creates a cubic network using OpenPNM's Cubic generator.
        Returns
        -------
        pn : openpnm.network.Cubic
            The generated cubic network.
        """
        pn = op.network.Cubic(shape=self.config.size, spacing=self.config.spacing)
        pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
        pn.regenerate_models()
        return pn

    def _create_imported(self):
        r"""
        Imports a network from Statoil format and applies geometry models.
        Returns
        -------
        pn : openpnm.network.GenericNetwork
            The imported and processed network.
        """
        project = op.io.network_from_statoil(path=self.config.path, prefix=self.config.prefix)
        pn = project.network
        pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
        pn.add_model(propname='pore.cluster_number', model=op.models.network.cluster_number)
        pn.add_model(propname='pore.cluster_size', model=op.models.network.cluster_size)
        pn['pore.diameter'] = pn['pore.radius'] * 2
        pn['throat.diameter'] = pn['throat.radius'] * 2
        # Remove disconnected pores identified by network health check
        h = op.utils.check_network_health(pn)
        op.topotools.trim(network=pn, pores=h['disconnected_pores'])
        pn.regenerate_models()
        return pn
        
        
    def _calculate_permeability(self):
        pn = self.network
        phase = op.phase.Phase(network=pn)
        phase['pore.viscosity']=1.0
        phase.add_model_collection(op.models.collections.physics.basic)
        phase.regenerate_models()
        inlet = pn.pores('left')
        outlet = pn.pores('right')
        
        flow = op.algorithms.StokesFlow(network=pn, phase=phase)
        flow.set_value_BC(pores=inlet, values=1)
        flow.set_value_BC(pores=outlet, values=0)
        flow.run()
        Q = flow.rate(pores=inlet, mode='group')[0]
        L = op.topotools.get_domain_length(pn, inlets=inlet, outlets=outlet)
        A = op.topotools.get_domain_area(pn, inlets=inlet, outlets=outlet)
        # K = Q * L * mu / (A * Delta_P) # mu and Delta_P were assumed to be 1.
        K = Q * L / A
        print(f'The value of K is: {K/0.98e-12*1000:.2f} mD')
        return K/0.98e-12*1000
    
    def _calculate_porosity(self):
        pn = self.network
        inlet = pn.pores('left')
        outlet = pn.pores('right')
        Vol_void = np.sum(pn['pore.volume'])+np.sum(pn['throat.volume'])
        A = op.topotools.get_domain_area(pn, inlets=inlet, outlets=outlet)
        L = op.topotools.get_domain_length(pn, inlets=inlet, outlets=outlet)
        Vol_bulk = A * L
        Poro = Vol_void / Vol_bulk
        return Poro