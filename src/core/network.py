from utils.config_parser import NetworkType
from enum import Enum
import numpy as np

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
        from openpnm.network import Cubic
        Lc = 1e-6
        pn = Cubic(shape=self.config.size, spacing=self.config.spacing)
        pn['pore.diameter'] = np.random.rand(pn.Np)*Lc
        pn['throat.diameter'] = np.random.rand(pn.Nt)*Lc
        
        return pn

    def _create_imported(self):
        from openpnm.io import network_from_statoil
        project = network_from_statoil(path = self.config.path, prefix = self.config.prefix)
        pn = project.network
        pn['pore.diameter'] = pn['pore.radius']*2
        pn['throat.diameter'] = pn['throat.radius']*2
        return pn
        