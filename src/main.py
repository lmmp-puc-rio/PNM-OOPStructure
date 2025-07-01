from utils.config_parser import ConfigParser
from core.network import Network
from core.phases import Phases

json_file = 'data/base.json'

cfg = ConfigParser.from_file(json_file)

pn = Network(config = cfg)
phases = Phases(network = pn, config = cfg)
