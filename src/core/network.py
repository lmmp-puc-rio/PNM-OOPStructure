from utils.config_parser import NetworkType
import numpy as np
import openpnm as op
import skimage.io as sc
import porespy as ps

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
        elif self.config.type == NetworkType.IMAGE:
            return self._create_image()
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
        
    def _create_image(self):
        r"""
        Creates a network from an image file.
        """
        im = sc.imread(self.config.file, as_gray=True)
        
        snow_params = {}
        for param_name in ['voxel_size', 'r_max', 'accuracy', 'sigma','boundary_width']:
            param_value = self.config.properties.get(param_name)
            if param_value is not None:
                snow_params[param_name] = param_value
        
        snow= ps.networks.snow2(im.T, **snow_params)
        pn = op.io.network_from_porespy(snow.network)
        h = op.utils.check_network_health(pn)
        op.topotools.trim(network=pn, pores=h['disconnected_pores'])
        pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
        pn['pore.diameter'] = pn["pore.inscribed_diameter"]
        pn['throat.diameter'] = pn["throat.inscribed_diameter"]
        pn['pore.inlets'] = pn["pore.xmin"]
        pn['pore.outlets'] = pn["pore.xmax"]
        pn['throat.spacing'] = pn['throat.total_length']
        pn.regenerate_models()
        return pn