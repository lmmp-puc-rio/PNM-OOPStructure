from utils.config_parser import ConfigParser
from core.network import Network
from core.phases import Phases
from core.algorithms import AlgorithmManager
from core.postprocessing import PostProcessing
import os
from utils.figures.media_utils import make_video, save_images_side_by_side

base_path = os.path.dirname(__file__)
json_file = 'data/base.json'

cfg = ConfigParser.from_file(json_file)

pn = Network(config = cfg)
phases = Phases(network = pn, config = cfg)
manager = AlgorithmManager(pn, phases, cfg)
results = manager.run_all()