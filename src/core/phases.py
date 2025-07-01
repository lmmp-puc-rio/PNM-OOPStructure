from utils.config_parser import PhaseModel
import numpy as np
import openpnm as op

class Phases:
    def __init__(self, network, config):
        self.config = config.phases
        self.network = network
        self.phases = dict()
        for dict_phase in self.config:
            self.phases[dict_phase.name] = (self._create_phase(dict_phase))
        
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
        phase.regenerate_models()
        
        for prop in raw.properties.keys():
            phase[prop] = raw.properties[prop]
        
        return phase
    
    def regenerate_models(self,name = None):
        props_by_name = self._get_default_properties()
        if name is None:
            for key in props_by_name.keys():
                self.phases[key].regenerate_models(exclude=props_by_name[key].keys())
        else:
            self.phases[name].regenerate_models(exclude=props_by_name[name].keys())
                  
        
    def _get_default_properties(self):
        return {phase.name: phase.properties for phase in self.config}