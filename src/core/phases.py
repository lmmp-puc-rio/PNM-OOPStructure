from utils.config_parser import PhaseModel
import openpnm as op

class Phases:
    def __init__(self, network, config):
        self.config = config.phases
        self.network = network
        self.phases = []
        for dict_phase in self.config:
            self.phases.append(
                dict(name = dict_phase.name,
                     model = self._create_phase(dict_phase),
                     color = dict_phase.color)) 
            
        
    def _create_phase(self, raw: dict):
        if raw.model == PhaseModel.WATER:
            phase = op.phase.Water(network = self.network.network, name = raw.name)
            phase.add_model_collection(op.models.collections.phase.water)
        elif raw.model == PhaseModel.AIR:
            phase = op.phase.Air(network = self.network.network, name = raw.name)
            phase.add_model_collection(op.models.collections.phase.air)
        else:
            raise ValueError(f"PhaseModel: {raw.model}")
        
        phase.add_model_collection(op.models.collections.physics.basic)
        
        for prop in raw.properties.keys():
            phase.add_model(propname = prop,
                            model = op.models.misc.constant,
                            value = raw.properties[prop])
            
        phase.regenerate_models()
        
        return phase
    
    def get_model(self, name):
        return next(p["model"] for p in self.phases if p["name"] == name)