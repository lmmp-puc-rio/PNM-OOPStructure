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

new_inlet_pores = [0,1,3,7,11,14,23,29,36,38,45,2,4,10,13,26,37,734,733,732]
pn.set_inlet_outlet_pores(inlet_pores=new_inlet_pores)
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