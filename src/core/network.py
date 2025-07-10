from utils.config_parser import NetworkType
import numpy as np
import openpnm as op

class Network:
    def __init__(self, config):
        self.config = config.network
        self.project_name = getattr(self.config, 'project_name', 'project')
        np.random.seed(self.config.seed)
        self.network = self._create_network()
        self.dim = '2D'
        if len(np.unique(self.network['pore.coords'].T[2])) > 1:
            self.dim = '3D'

    def _create_network(self):
        if self.config.type == NetworkType.CUBIC:
            return self._create_cubic()
        elif self.config.type == NetworkType.IMPORTED:
            return self._create_imported()
        else:
            raise ValueError(f"NetworkType: {self.config.type}")

    def _create_cubic(self): 
        pn = op.network.Cubic(shape=self.config.size, spacing=self.config.spacing)
        pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
        pn.regenerate_models()
        return pn

    def _create_imported(self):
        project = op.io.network_from_statoil(path = self.config.path, prefix = self.config.prefix)
        pn = project.network
        pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
        pn.add_model(propname='pore.cluster_number',
             model=op.models.network.cluster_number)
        pn.add_model(propname='pore.cluster_size',
                    model=op.models.network.cluster_size)
        pn['pore.diameter'] = pn['pore.radius']*2
        pn['throat.diameter'] = pn['throat.radius']*2
        Ps = pn['pore.cluster_size']< 6004
        op.topotools.trim(network=pn, pores=Ps)
        
        pn.regenerate_models()
        return pn
        