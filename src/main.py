from utils.config_parser import ConfigParser
from core.network import Network

json_file = 'data/base.json'

cfg = ConfigParser.from_file(json_file)

pn = Network(cfg)
