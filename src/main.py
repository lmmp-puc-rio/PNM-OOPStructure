from utils.config_parser import ConfigParser
from core.network import Network
from core.phases import Phases
from core.algorithm import Algorithm
from core.postprocessing import PostProcessing
import os
from utils.figures.media_utils import make_video, save_images_side_by_side

base_path = os.path.dirname(__file__)
json_file = 'data/base.json'

cfg = ConfigParser.from_file(json_file)

pn = Network(config = cfg)
print(pn._calculate_permeability())
print(pn._calculate_porosity())
# phases = Phases(network = pn, config = cfg)
# algorithm = Algorithm(network = pn, phases = phases,config = cfg)
# algorithm.run()

# algorithm._calculate_permeability(algorithm.algorithm[0])
# algorithm._calculate_porosity(algorithm.algorithm[0])
# print('a')

# post.plot_network()
# invasionPath = post.make_frames_type('invasion')
# clustersPath = post.make_frames_type('clusters')
# make_video(frames_path=invasionPath, fps=2, output_file=os.path.join(post.video_path, 'invasion.mp4'))
# make_video(frames_path=clustersPath, fps=2, output_file=os.path.join(post.video_path, 'clusters.mp4'))
# rel1 = post.plot_relative_permeability(alg=algorithm.algorithm[0]['algorithm'], Snwp_num=20)
# rel2 = post.plot_relative_permeability(alg=algorithm.algorithm[1]['algorithm'], Snwp_num=20)
# save_images_side_by_side(rel1, rel2, os.path.join(post.graph_path, 'relative_permeability_side_by_side.png'))