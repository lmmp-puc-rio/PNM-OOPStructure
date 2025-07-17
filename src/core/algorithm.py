from utils.config_parser import AlgorithmType
import openpnm as op
import numpy as np

class Algorithm:
    def __init__(self, network, phases, config):
        self.config = config.algorithm
        self.network = network
        self.phases = phases
        self.algorithm = []
        for dict_algorithm in self.config:
            phase_dict = next(phase for phase in self.phases.phases if phase["name"] == dict_algorithm.phase)
            self.algorithm.append(
                dict(
                    name=dict_algorithm.name,
                    algorithm=self._create_algorithm(dict_algorithm),
                    phase=phase_dict,
                    config=dict_algorithm
                )
            )
            
    def _create_algorithm(self, raw: dict):
        name = raw.name
        phase = self.phases.get_model(raw.phase)
        pn = self.network.network
        inlet = pn.pores(raw.inlet)
        if raw.type == AlgorithmType.DRAINAGE:
            alg = op.algorithms.Drainage(network = pn, 
                                        phase = phase,
                                        name = name)
            alg.set_inlet_BC(pores = inlet, mode='overwrite')
            if raw.outlet is not None:
                outlet = pn.pores(raw.outlet)
                alg.set_outlet_BC(pores = outlet, mode='overwrite')
                
        elif raw.type == AlgorithmType.STOKES:
            conductance='throat.non_newtonian_conductance'
            self.phases.add_non_newtonian_conductance_model(phase)
            phase.regenerate_models()
            alg = op.algorithms.StokesFlow(network=pn, phase=phase)
            alg.settings._update({'conductance': conductance})
        return alg
    
    def run(self):
        pore_trapped = np.full(self.network.network.Np,False)
        throat_trapped = np.full(self.network.network.Nt,False)
        
        for alg in self.algorithm:
            if alg['config'].type == AlgorithmType.DRAINAGE:
                self._run_drainage(alg,pore_trapped, throat_trapped)
                pore_trapped = alg['algorithm']['pore.trapped'].copy()
                throat_trapped = alg['algorithm']['throat.trapped'].copy()
            elif alg['config'].type == AlgorithmType.STOKES:
                self._run_stokes(alg)
        return
    
    def get_model(self, name):
        return next(p for p in self.algorithm if p.name == name)
           
    def _run_drainage(self, alg, pore_trapped, throat_trapped):
        algorithm = alg['algorithm']
            
        algorithm['pore.ic_invaded'] =  pore_trapped.copy()
        algorithm['throat.ic_invaded'] =  throat_trapped.copy()
        #Setting initial conditions
        if np.any(pore_trapped):
            algorithm['pore.invaded'] = pore_trapped.copy()
        if np.any(throat_trapped):
            algorithm['throat.invaded'] = throat_trapped.copy()
            
        #calculate pressures
        phase_model = alg['phase']['model']
        p_max = phase_model['throat.entry_pressure'].max()
        p_min = phase_model['throat.entry_pressure'].min()
        samples = alg['config'].pressures
        x = np.linspace(0,1,samples)
        pressures = p_min * (p_max/p_min) ** x
        
        algorithm.run(pressures = pressures)
        return
    
    def _run_stokes(self,alg):
        algorithm = alg['algorithm']
        pn = self.network.network
        P_min = alg['config'].initial_pressure
        P_max = alg['config'].final_pressure
        steps = alg['config'].pressures
        inlet = alg['config'].inlet
        outlet = alg['config'].outlet
        p_sequence = np.linspace(P_min, P_max, steps)
        inlet = pn.pores(inlet)
        outlet = pn.pores(outlet)
        Q= []
        for p in p_sequence:
            algorithm.set_value_BC(pores=inlet, values=p,mode='overwrite')
            algorithm.set_value_BC(pores=outlet, values=0,mode='overwrite')
            algorithm.run()
            Q.append(algorithm.rate(pores=inlet, mode='group')[0])
        alg['results'] = {'pressure': p_sequence, 'flow_rate': Q}
        return