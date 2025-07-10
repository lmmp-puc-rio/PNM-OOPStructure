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
                     model = self._create_phase_model(dict_phase),
                     color = dict_phase.color)) 
            
        
    def _create_phase_model(self, raw: dict):
        if raw.model == PhaseModel.WATER:
            phase_model = op.phase.Water(network = self.network.network, name = raw.name)
            phase_model.add_model_collection(op.models.collections.phase.water)
            
        elif raw.model == PhaseModel.AIR:
            phase_model = op.phase.Air(network = self.network.network, name = raw.name)
            phase_model.add_model_collection(op.models.collections.phase.air)
        else:
            raise ValueError(f"PhaseModel: {raw.model}")
        
        phase_model.add_model_collection(op.models.collections.physics.basic)
        
        for prop in raw.properties.keys():
            phase_model.add_model(propname = prop,
                            model = op.models.misc.constant,
                            value = raw.properties[prop])
            
        phase_model.regenerate_models()
        
        return phase_model
    
    def get_model(self, name):
        return next(phase["model"] for phase in self.phases if phase["name"] == name)
    
    def get_wetting_phase(self):
        """Return the model of the wetting phase (contact angle < 90)."""
        for phase in self.phases:
            model = phase['model']
            if model['pore.contact_angle'][0] < 90:
                return phase
        return None

    def get_non_wetting_phase(self):
        """Return the model of the non-wetting phase (contact angle >= 90)."""
        for phase in self.phases:
            model = phase['model']
            if model['pore.contact_angle'][0] >= 90:
                return phase
        return None
    
    def add_conduit_conductance_model(self, phase_model):
            """Adds the conduit conductance model to the given phase model."""
            model_mp_cond = op.models.physics.multiphase.conduit_conductance
            phase_model.add_model(
                model=model_mp_cond,
                propname='throat.conduit_hydraulic_conductance',
                throat_conductance='throat.hydraulic_conductance',
                mode='medium',
                regen_mode='deferred'
            )