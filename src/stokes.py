from utils.config_parser import ConfigParser
from core.network import Network
from core.phases import Phases
from core.algorithms import AlgorithmManager
from core.postprocessing import PostProcessingManager
import os

base_path = os.path.dirname(__file__)
json_file = 'data/non_newtonianStokes.json'

cfg = ConfigParser.from_file(json_file)
pn = Network(config=cfg)
phases = Phases(network=pn, config=cfg)

manager = AlgorithmManager(pn, phases, cfg)
results = manager.run_all()

postproc = PostProcessingManager(manager, base_path)
postproc.plot_network(lwidth=3, msize=100)
postproc.stokes_processor.plot_absolute_permeability('stokesSimulation')
postproc.stokes_processor.plot_apparent_viscosity('stokesSimulation')
postproc.stokes_processor.plot_pressure_field('stokesSimulation')