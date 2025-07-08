from utils.config_parser import ConfigParser
from core.network import Network
from core.phases import Phases
from core.algorithm import Algorithm
from core.postprocessing import PostProcessing
import os

base_path = os.path.dirname(__file__)
json_file = 'data/base.json'

cfg = ConfigParser.from_file(json_file)

pn = Network(config = cfg)
phases = Phases(network = pn, config = cfg)
algorithm = Algorithm(network = pn, phases = phases,config = cfg)
algorithm.run()

post = PostProcessing(algorithm=algorithm, base_path=base_path)

post.plot_network()
post.make_frames_side_by_side()
post.make_video(frames_path=post.frames_side_by_side, fps=2,output_file=os.path.join(post.video_path, 'side_by_side.mp4'))
post.make_video(frames_path=post.invasion_path, fps=2,output_file=os.path.join(post.video_path, 'invasion.mp4'))
