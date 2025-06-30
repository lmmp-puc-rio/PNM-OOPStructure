from utils.config_parser import NetworkType
import numpy as np
import openpnm as op

class Network:
    def __init__(self, config):
        self.config = config.network
        np.random.seed(self.config.seed)
        self.network = self._create_network()

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
        pn['pore.diameter'] = pn['pore.radius']*2
        pn['throat.diameter'] = pn['throat.radius']*2
        return pn
        