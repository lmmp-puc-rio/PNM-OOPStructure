import openpnm as op
import numpy as np

class Algorithm:
    def __init__(self, network, phases, config):
        self.config = config.algorithm
        self.network = network
        self.phases = phases
        self.algorithm = []
        for dict_algorithm in self.config:
            self.algorithm.append(self._create_algorithm(dict_algorithm))
            
    def _create_algorithm(self, raw: dict):
        alg = op.algorithms.Drainage(network = self.network.network, 
                                     phase = self.phases.get_model(raw.phase),
                                     name = raw.name)
        inlet = self.network.network.pores(raw.inlet)
        alg.set_inlet_BC(pores = inlet, mode='overwrite')
        if raw.outlet is not None:
            outlet = self.network.network.pores(raw.outlet)
            alg.set_outlet_BC(pores = outlet, mode='overwrite')
        return alg
    
    def run(self):    
        pore_trapped = np.full(self.network.network.Np,False)
        throat_trapped = np.full(self.network.network.Nt,False)
        
        for alg in self.algorithm:
            #Setting initial conditions
            if np.any(pore_trapped):
                alg['pore.invaded'] = pore_trapped.copy()
            if np.any(throat_trapped):
                alg['throat.invaded'] = throat_trapped.copy()
                
            alg['pore.ic_invaded'] =  pore_trapped.copy()
            alg['throat.ic_invaded'] =  throat_trapped.copy()
            #calculate pressures
            phase = alg.settings.phase
            phase_model = self.phases.get_model(phase)
            p_max = phase_model['throat.entry_pressure'].max()
            p_min = phase_model['throat.entry_pressure'].min()
            samples = next(p.pressures for p in self.config if p.name == alg.name)
            x = np.linspace(0,1,samples)
            pressures = p_min * (p_max/p_min) ** x
            
            alg.run(pressures = pressures)
            
            pore_trapped = alg['pore.trapped'].copy()
            throat_trapped = alg['throat.trapped'].copy()
        return
    
    def get_model(self, name):
        return next(p for p in self.algorithm if p.name == name)