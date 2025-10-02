from utils.config_parser import ConfigParser
from core.network import Network
from core.phases import Phases
from core.algorithms import AlgorithmManager
from core.postprocessing import PostProcessingManager
import os
from utils.figures.media_utils import make_video, save_images_side_by_side

base_path = os.path.dirname(__file__)
json_file = 'data/image.json'

cfg = ConfigParser.from_file(json_file)
pn = Network(config=cfg)
pn.redefine_throat_radius(mean=5*10**-5)

pn.calculate_permeability()

phases = Phases(network=pn, config=cfg)

manager = AlgorithmManager(pn, phases, cfg)
results = manager.run_all()

postproc = PostProcessingManager(manager, base_path)
postproc.plot_network_tutorial()
postproc.plot_network()
postproc.drainage_processor.plot_capillary_pressure_curve('drainageSimulation')
postproc.drainage_processor.make_invasion_frames('drainageSimulation')
postproc.drainage_processor.make_invasion_frames('imbibitionSimulation')
postproc.drainage_processor.make_clusters_frames('drainageSimulation')
postproc.drainage_processor.make_clusters_frames('imbibitionSimulation')
postproc.drainage_processor.plot_relative_permeability('drainageSimulation')
postproc.drainage_processor.plot_relative_permeability('imbibitionSimulation')