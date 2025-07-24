from utils.config_parser import ConfigParser
from core.network import Network
from core.phases import Phases
from core.algorithm import Algorithm
from core.postprocessing import PostProcessing
import os

base_path = os.path.dirname(__file__)
json_file = 'data/non_newtonianStokes.json'

cfg = ConfigParser.from_file(json_file)

pn = Network(config = cfg)


import openpnm as op
import numpy as np

network = pn.network

network.add_model(propname='pore.volume',
             model=op.models.geometry.pore_volume.sphere)
network.add_model(propname='throat.length',
             model=op.models.geometry.throat_length.spheres_and_cylinders)
network.add_model(propname='throat.total_volume',
             model=op.models.geometry.throat_volume.cylinder)
network.add_model(propname='throat.lens_volume', 
            model=op.models.geometry.throat_volume.lens)
network.add_model(propname='throat.volume', 
             model=op.models.misc.difference,
             props=['throat.total_volume', 'throat.lens_volume'])

network.models['throat.diameter@all']['factor'] = 0.7

np.random.seed(0)  # Set the state of the random number generator to "0"
network['pore.seed'] = np.random.rand(network.Np)

import scipy.stats as spst
dst = spst.weibull_min(c=1.9, loc=1e-7, scale=20e-6)
network['pore.diameter'] = dst.rvs(network.Np)

f = op.models.geometry.pore_seed.random
network.add_model(propname='pore.seed',
             model=f,
             num_range=[0.0, 0.7])
network.add_model(propname='pore.diameter',
             model=op.models.geometry.pore_size.generic_distribution,
             func=dst,
             seeds='pore.seed')

network.add_model(propname='throat.diameter_1', 
             model=op.models.misc.from_neighbor_pores,
             prop='pore.diameter',
             mode='min')
network.add_model(propname='throat.seed', 
             model=op.models.misc.from_neighbor_pores,
             prop='pore.seed',
             mode='min')
network.add_model(propname='throat.diameter_2',
             model=op.models.geometry.throat_size.generic_distribution,
             func=dst)

network.add_model(propname='throat.diameter',
             model=op.models.misc.scaled,
             prop='throat.diameter_1',  # This could also be 'throat.diameter_2'
             factor=0.7,  # This could be 1.0 if no scaling is desired
)

network.regenerate_models()
print(pn._calculate_permeability())
print(pn._calculate_porosity())





phases = Phases(network = pn, config = cfg)
algorithm = Algorithm(network = pn, phases = phases,config = cfg)
algorithm.run()

post = PostProcessing(algorithm=algorithm, base_path=base_path)
post.plot_network()
post.plot_absolute_permeability(alg=algorithm.algorithm[0])
post.plot_diameter_distribution()
