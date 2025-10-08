r"""
Drainage post-processor for analyzing two-phase flow results.

This module provides specialized post-processing capabilities for drainage
and imbibition simulations.
"""

import os
import numpy as np
import openpnm as op
from itertools import count
from .base_postprocessor import BasePostProcessor
from utils.plots.plotter import Plotter2D, Plotter3D


class DrainagePostProcessor(BasePostProcessor):
    r"""
    Post-processor for drainage/imbibition simulation results.
    
    This class provides specialized methods for analyzing and visualizing
    two-phase flow results.
    
    Parameters
    ----------
    algorithm_manager : AlgorithmManager
        The algorithm manager containing drainage results
    base_path : str
        Base directory path for saving outputs
    """
    
    def __init__(self, algorithm_manager, base_path):
        super().__init__(algorithm_manager, base_path)
        
    def make_invasion_frames(self, algorithm_name, lwidth=3, msize=100):
        r"""
        Generate frames showing invasion progression.
        
        Parameters
        ----------
        algorithm_name : str
            Name of the drainage algorithm to visualize
        lwidth : float, default 3
            Maximum line width for visualization
        msize : float, default 100
            Maximum marker size for visualization
            
        Returns
        -------
        frame_path : str
            Path to directory containing generated frames
        """
        def title_func(alg, seq):
            p_kpa = alg['throat.invasion_pressure'][
                alg['throat.invasion_sequence'] == seq
            ].max() / 1000
            return f'Pressure = {p_kpa:.2f} kPa'
            
        def draw_func(**kwargs):
            self._draw_invasion(**kwargs)
                            
        return self._make_frames(
            algorithm_name, draw_func, title_func, 'invasion_frames', lwidth, msize
        )
        
    def make_clusters_frames(self, algorithm_name, lwidth=3, msize=100):
        r"""
        Generate frames showing cluster formation and trapping.
        
        Parameters
        ----------
        algorithm_name : str
            Name of the drainage algorithm to visualize
        lwidth : float, default 3
            Maximum line width for visualization
        msize : float, default 100
            Maximum marker size for visualization
            
        Returns
        -------
        frame_path : str
            Path to directory containing generated frames
        """
        def title_func(alg, seq):
            return 'Clusters / Trapping'
            
        def draw_func(**kwargs):
            self._draw_clusters(
                kwargs['ax'], kwargs['pn'], kwargs['alg'], 
                kwargs['sequence'], kwargs['linewidth'], kwargs['markersize']
            )
            
        return self._make_frames(
            algorithm_name, draw_func, title_func, 'clusters_frames', lwidth, msize
        )
        
    def plot_relative_permeability(self, algorithm_name, Snwp_num=20, output_file=None):
        r"""
        Plot relative permeability curves for both phases.
        
        Parameters
        ----------
        algorithm_name : str
            Name of the drainage algorithm
        Snwp_num : int, default 20
            Number of saturation points to calculate
        output_file : str, optional
            Custom output file path
            
        Returns
        -------
        output_file : str
            Path to the saved plot
        """
        kr_dict = self._load_data()
        kr_wp, kr_nwp = kr_dict[algorithm_name]
        
        alg_dict = self.algorithm_manager.get_algorithm(algorithm_name)
        algorithm = alg_dict['algorithm']
        pn = self.algorithm_manager.network.network
        
        wp  = self.algorithm_manager.phases.get_wetting_phase()
        nwp = self.algorithm_manager.phases.get_non_wetting_phase()
        wp_model = wp['model']
        nwp_model = nwp['model']
        
        self.algorithm_manager.phases.add_conduit_conductance_model(wp_model)
        self.algorithm_manager.phases.add_conduit_conductance_model(nwp_model)
        
        inlet = pn.pores('inlet')
        outlet = pn.pores('outlet')
        
        Snwparr, relperm_nwp, relperm_wp = self._calculate_relative_permeability(
            pn, wp_model, nwp_model, algorithm.algorithm, inlet, outlet, Snwp_num
        )
        relperm_nwp = np.array(relperm_nwp)/np.max(relperm_nwp)
        
        plotter = Plotter2D(
            layout='relative_permeability', 
            title=f'Relative Permeability {algorithm_name}'
        )
        ax = plotter.ax
        ax.plot(100-np.array(Snwparr), relperm_nwp, '-o', label='Kr_nwp', color=nwp['color'])
        ax.plot(100-np.array(Snwparr), relperm_wp,  '-*', label='Kr_wp',  color=wp['color'])

        ax.scatter(100*kr_wp[:,0], kr_wp[:,1])
        ax.scatter(100*kr_nwp[:,0], kr_nwp[:,1])
        
        plotter.apply_layout()
        output_file = output_file or os.path.join(
            self.graph_path, f'RP_{algorithm_name}.png'
        )
        plotter.save(output_file)
        return output_file

    def plot_hysteresis(self, algorithm_names, Snwp_num=20, output_file=None):
        r"""
        Plot relative permeability curves for both phases.
        
        Parameters
        ----------
        algorithm_name : str
            Name of the drainage algorithm
        Snwp_num : int, default 20
            Number of saturation points to calculate
        output_file : str, optional
            Custom output file path
            
        Returns
        -------
        output_file : str
            Path to the saved plot
        """
        wetting_phase = {'name': 'Wetting Phase','color': ['#0000ff','#000066'],'Snwp': [],'relperm': [], 'algorithm_name': []}
        non_wetting_phase = {'name': 'Non-Wetting Phase','color': ['#ff0000','#660000'],'Snwp': [],'relperm': [], 'algorithm_name': []}
        for algorithm_name in algorithm_names:
            alg_dict = self.algorithm_manager.get_algorithm(algorithm_name)        
            algorithm = alg_dict['algorithm']
            pn = self.algorithm_manager.network.network
        
            wp = self.algorithm_manager.phases.get_wetting_phase()
            nwp = self.algorithm_manager.phases.get_non_wetting_phase()
            wp_model = wp['model']
            nwp_model = nwp['model']
            
            self.algorithm_manager.phases.add_conduit_conductance_model(wp_model)
            self.algorithm_manager.phases.add_conduit_conductance_model(nwp_model)
            
            inlet = pn.pores('inlet')
            outlet = pn.pores('outlet')
            
            Snwparr_alg, relperm_nwp_alg, relperm_wp_alg = self._calculate_relative_permeability(
                pn, wp_model, nwp_model, algorithm.algorithm, inlet, outlet, Snwp_num
            )
            wetting_phase['Snwp'].append(Snwparr_alg)
            wetting_phase['relperm'].append(relperm_wp_alg)
            wetting_phase['algorithm_name'].append(algorithm_name)
            non_wetting_phase['Snwp'].append(Snwparr_alg)
            non_wetting_phase['relperm'].append(relperm_nwp_alg)
            non_wetting_phase['algorithm_name'].append(algorithm_name)

        for phase in [wetting_phase, non_wetting_phase]:
            plotter = Plotter2D(
                layout='relative_permeability', 
                title=f"Relative Permeability {phase['name']}"
            )
            ax = plotter.ax
            ax.plot(phase['Snwp'][0], phase['relperm'][0], '->', label=f"Kr_{phase['algorithm_name'][0]}", 
                    color=phase['color'][0],markerfacecolor='#ffffff')
            ax.plot(phase['Snwp'][1], phase['relperm'][1], '--<', label=f"Kr_{phase['algorithm_name'][1]}", 
                    color=phase['color'][1],markerfacecolor='#ffffff')
            # Set x-axis tick label font size
            ax.tick_params(axis='x', labelsize=18) 

            # Set y-axis tick label font size
            ax.tick_params(axis='y', labelsize=18) 
            plotter.apply_layout()
            output_file = os.path.join(
                self.graph_path, f"hysteresis_{phase['name']}.png"
            )
            plotter.save(output_file)
        return output_file
        
    def plot_capillary_pressure_curve(self, algorithm_name, output_file=None):
        r"""
        Plot capillary pressure curve.
        
        Parameters
        ----------
        algorithm_name : str
            Name of the drainage algorithm
        output_file : str, optional
            Custom output file path
            
        Returns
        -------
        output_file : str
            Path to the saved plot
        """
        alg_dict = self.algorithm_manager.get_algorithm(algorithm_name)
        algorithm = alg_dict['algorithm']
        phase_dict = alg_dict['phase']
        
        data_pc = algorithm.pc_curve()
        
        plotter = Plotter2D(
            layout='capillary_pressure', 
            title=f'Capillary Pressure Curve {algorithm_name}'
        )
        ax = plotter.ax
        ax.plot(data_pc.pc, data_pc.snwp*100, color=phase_dict['color'])
        
        plotter.apply_layout()
        output_file = output_file or os.path.join(
            self.graph_path, f'capillary_pressure_{algorithm_name}.png'
        )
        plotter.save(output_file)
        return output_file
        
    def _make_frames(self, algorithm_name, draw_func, title_func, frame_subdir, lwidth, msize):
        r"""Generate animation frames for drainage visualization."""
        alg_dict = self.algorithm_manager.get_algorithm(algorithm_name)
        algorithm = alg_dict['algorithm']
        pn = self.algorithm_manager.network.network
        phases = self.algorithm_manager.phases.phases
        dim = self.algorithm_manager.network.dim
        
        linewidth = pn['throat.diameter'] / pn['throat.diameter'].max() * lwidth
        markersize = pn['pore.diameter'] / pn['pore.diameter'].max() * msize
        
        algorithm_frame_subdir = f"{frame_subdir}_{algorithm_name}"
        frame_path = os.path.join(self.frame_path, algorithm_frame_subdir)
        os.makedirs(frame_path, exist_ok=True)
        
        #TODO include inital phase in JSON file 
        inv_phase = algorithm.algorithm.settings.phase
        inv_color = alg_dict['phase']['color']
        not_inv_color = next(p['color'] for p in phases if p['name'] != inv_phase)
        
        alg = algorithm.algorithm
        throats_ic = pn.Ts[alg['throat.ic_invaded']].copy()
        pores_ic = np.union1d(pn.Ps[alg['pore.ic_invaded']].copy(), pn.pores('inlet'))
        invasion_sequence = np.unique(
            alg['throat.invasion_sequence'][np.isfinite(alg['throat.invasion_sequence'])]
        )
        
        plotter_cls = Plotter3D if dim == '3D' else Plotter2D
        _frame_id = count()
        
        title = 'Initial State '
        plotter = plotter_cls(
                layout=f'invasion_{"3d" if dim=="3D" else "2d"}', 
                title=title
            )
        ax = plotter.ax
        
        self._plot_pores_and_throats(
            pn, pores=pores_ic, throats=throats_ic,
            color=inv_color, alpha=0.8, 
            markersize=markersize, linewidth=linewidth, ax=ax
        )
        
        self._plot_pores_and_throats(
            pn, pores=np.setdiff1d(pn.Ps, pores_ic), throats=np.setdiff1d(pn.Ts, throats_ic),
            color=not_inv_color, alpha=0.8, 
            markersize=markersize, linewidth=linewidth, ax=ax
        )
        idx = next(_frame_id)
        plotter.apply_layout()
        plotter.save(os.path.join(frame_path, f'{algorithm_frame_subdir}_{idx:04d}.png'))
        
        for seq in invasion_sequence:
            title = title_func(alg, seq)
            plotter = plotter_cls(
                layout=f'invasion_{"3d" if dim=="3D" else "2d"}', 
                title=title
            )
            ax = plotter.ax
            draw_func(
                ax=ax, pn=pn, alg=alg, sequence=seq, 
                inv_color=inv_color, not_inv_color=not_inv_color,
                linewidth=linewidth, markersize=markersize, 
                throats_ic=throats_ic, pores_ic=pores_ic
            )
            idx = next(_frame_id)
            plotter.apply_layout()
            plotter.save(os.path.join(frame_path, f'{algorithm_frame_subdir}_{idx:04d}.png'))
            
        for j in np.arange(0, 1, 0.1):
            plotter = plotter_cls(
                layout=f'invasion_{"3d" if dim=="3D" else "2d"}', 
                title=title
            )
            ax = plotter.ax
            draw_func(
                ax=ax, pn=pn, alg=alg, sequence=seq,
                inv_color=inv_color, not_inv_color=not_inv_color,
                linewidth=linewidth, markersize=markersize, 
                throats_ic=throats_ic, pores_ic=pores_ic,
                alpha_inv=1-j, alpha_not_inv=j, alpha_ic=1-j
            )
            idx = next(_frame_id)
            plotter.apply_layout()
            plotter.save(os.path.join(frame_path, f'{algorithm_frame_subdir}_{idx:04d}.png'))

        # Add 3D rotation frames
        if dim == '3D':
            n_frames = 36
            azim_step = 10
            azim_0 = -60
            for l in range(n_frames):
                plotter = Plotter3D(layout='invasion_3d', title=title)
                ax = plotter.ax
                draw_func(
                    ax=ax, pn=pn, alg=alg, sequence=seq,
                    inv_color=inv_color, not_inv_color=not_inv_color,
                    linewidth=linewidth, markersize=markersize, 
                    throats_ic=throats_ic, pores_ic=pores_ic,
                    alpha_inv=0, alpha_not_inv=1, alpha_ic=0
                )
                plotter.layout.update(azim=azim_0 + l * azim_step)
                idx = next(_frame_id)
                plotter.apply_layout()
                plotter.save(os.path.join(frame_path, f'{algorithm_frame_subdir}_{idx:04d}.png'))
                
        return frame_path
        
    def _draw_invasion(self, ax, pn, alg, sequence, inv_color, not_inv_color,
                      linewidth, markersize, throats_ic, pores_ic, 
                      alpha_inv=0.8, alpha_not_inv=0.8, alpha_ic=0.4):
        r"""Draw invasion state at specific sequence."""
        self._plot_pores_and_throats(
            pn, pores=pores_ic, throats=throats_ic,
            color=inv_color, alpha=alpha_ic, 
            markersize=markersize, linewidth=linewidth, ax=ax
        )          
          
        mask_throat = alg['throat.invasion_sequence'] <= sequence
        mask_pore = alg['pore.invasion_sequence'] <= sequence
        new_throats = np.setdiff1d(pn.Ts[mask_throat], throats_ic)
        new_pores = np.setdiff1d(pn.Ps[mask_pore], pores_ic)
        still_not_t = np.setdiff1d(pn.Ts[~mask_throat], throats_ic)
        still_not_p = np.setdiff1d(pn.Ps[~mask_pore], pores_ic)
        
        inlet = pn.pores('inlet')
        new_pores = np.union1d(new_pores, inlet)
        still_not_p = np.setdiff1d(still_not_p, inlet)
        
        self._plot_pores_and_throats(
            pn, pores=new_pores, throats=new_throats,
            color=inv_color, alpha=alpha_inv, 
            markersize=markersize, linewidth=linewidth, ax=ax
        )
            
        self._plot_pores_and_throats(
            pn, pores=still_not_p, throats=still_not_t,
            color=not_inv_color, alpha=alpha_not_inv, 
            markersize=markersize, linewidth=linewidth, ax=ax
        )
    
    def _draw_clusters(self, ax, pn, alg, sequence, linewidth, markersize):
        r"""Draw cluster analysis and trapping visualization."""
        p = alg['throat.invasion_pressure'][
            alg['throat.invasion_sequence'] == sequence
        ].max()
        pseq = alg['pore.invasion_pressure']
        tseq = alg['throat.invasion_pressure']
        occupied = pseq > p
        s, b = op._skgraph.simulations.site_percolation(conns=pn.conns, occupied_sites=occupied)
        
        both_pores_invaded = ((pseq[alg.network.conns[:, 0]] <= p) & (pseq[alg.network.conns[:, 1]] <= p))
        same_cluster = s[alg.network.conns[:, 0]] == s[alg.network.conns[:, 1]]
        uninvaded_throat = tseq > p
        trap_condition = both_pores_invaded & same_cluster & uninvaded_throat
        trapped_throats = trap_condition
        
        clusters_out = np.unique(s[pn.pores('outlet')])
        Ts = pn.find_neighbor_throats(pores=s >= 0)
        b[Ts] = np.amax(s[pn.conns], axis=1)[Ts]
        trapped_pores = np.isin(s, clusters_out, invert=True) & (s >= 0)
        trapped_throats += np.isin(b, clusters_out, invert=True) & (b >= 0)

        self._plot_pores_and_throats(
            pn, pores=trapped_pores, linewidth=linewidth, markersize=markersize, 
            color_by=s[trapped_pores], ax=ax
        )
        self._plot_pores_and_throats(
            pn, throats=trapped_throats, linewidth=linewidth, markersize=markersize,  
            color_by=b[trapped_throats], ax=ax
        )
        
        mask_inv_p = pseq <= p
        mask_inv_t = alg['throat.invasion_pressure'] <= p
        self._plot_pores_and_throats(
            pn, pores=mask_inv_p, linewidth=linewidth, markersize=markersize, 
            c='k', ax=ax
        )
        self._plot_pores_and_throats(
            pn, throats=mask_inv_t, linewidth=linewidth, markersize=markersize, 
            c='k', linestyle='--', ax=ax
        )
        
        self._plot_pores_and_throats(
            pn, pores=pn.Ps, linewidth=linewidth, markersize=markersize,
            color='k', alpha=0.0, ax=ax
        )
        
    def _calculate_relative_permeability(self, pn, wp_model, nwp_model, algorithm, 
                                       inlet, outlet, Snwp_num):
        r"""Calculate relative permeability curves."""
        def update_occupancy_and_get_saturation(network, nwp, wp, alg, seq):
            inv_phase_name = alg.settings.phase
            non_wetting_phase = self.algorithm_manager.phases.get_non_wetting_phase()
            is_invading_nwp = (inv_phase_name == non_wetting_phase['name'])

            ic_pores = alg['pore.ic_invaded'].copy()
            ic_throats = alg['throat.ic_invaded'].copy()

            invaded_pores_seq = alg['pore.invasion_sequence'] <= seq
            invaded_throats_seq = alg['throat.invasion_sequence'] <= seq
            
            invaded_pores = invaded_pores_seq | ic_pores
            invaded_throats = invaded_throats_seq | ic_throats

            if is_invading_nwp:
                nwp['pore.occupancy'] = invaded_pores
                nwp['throat.occupancy'] = invaded_throats
                wp['pore.occupancy'] = ~invaded_pores
                wp['throat.occupancy'] = ~invaded_throats
                nw_sat_p = np.sum(network['pore.volume'][invaded_pores])
                nw_sat_t = np.sum(network['throat.volume'][invaded_throats])
            else:
                wp['pore.occupancy'] = invaded_pores
                wp['throat.occupancy'] = invaded_throats
                nwp['pore.occupancy'] = ~invaded_pores
                nwp['throat.occupancy'] = ~invaded_throats
                nw_sat_p = np.sum(network['pore.volume'][~invaded_pores])
                nw_sat_t = np.sum(network['throat.volume'][~invaded_throats])
                
            total_volume = network['pore.volume'].sum() + network['throat.volume'].sum()
            saturation = (nw_sat_p + nw_sat_t) / total_volume
            return saturation

        def Rate_calc(network, phase, inlet, outlet, conductance):
            phase.regenerate_models()
            St_p = op.algorithms.StokesFlow(network=network, phase=phase)
            St_p.settings._update({'conductance': conductance})
            St_p.set_value_BC(pores=inlet, values=1)
            St_p.set_value_BC(pores=outlet, values=0)
            St_p.run()
            val = np.abs(St_p.rate(pores=inlet, mode='group'))
            return val

        tmask = np.isfinite(algorithm['throat.invasion_sequence'])
        max_seq = np.max(algorithm['throat.invasion_sequence'][tmask])
        min_seq = np.min(algorithm['throat.invasion_sequence'][tmask])
        relperm_sequence = np.linspace(min_seq, max_seq, Snwp_num).astype(int)
        
        Snwparr, relperm_nwp, relperm_wp = [], [], []

        for i in relperm_sequence:
            sat = update_occupancy_and_get_saturation(pn, nwp_model, wp_model, algorithm, i)
            Snwparr.append(sat * 100)
            
            Rate_abs_nwp = Rate_calc(pn, nwp_model, inlet, outlet, 'throat.hydraulic_conductance')
            Rate_abs_wp  = Rate_calc(pn, wp_model,  inlet, outlet, 'throat.hydraulic_conductance')
            Rate_enwp    = Rate_calc(pn, nwp_model, inlet, outlet, 'throat.conduit_hydraulic_conductance')
            Rate_ewp     = Rate_calc(pn, wp_model,  inlet, outlet, 'throat.conduit_hydraulic_conductance')
            
            relperm_nwp.append(Rate_enwp / Rate_abs_nwp)
            relperm_wp.append(Rate_ewp / Rate_abs_wp)

        return Snwparr, relperm_nwp, relperm_wp
    
    def _load_data(self):
        kr_drain_wp = """0.23999999999999988, -0.0033333333333334103
        0.25993333333333335, 0.0033333333333334103
        0.28789999999999993, 0.006666666666666599
        0.33596666666666664, 0
        0.3679333333333332, 0.0033333333333334103
        0.32386666666666664, 0.010000000000000009
        0.31189999999999996, 0.006666666666666599
        0.3878666666666666, 0.010000000000000009
        0.41779999999999995, 0.016666666666666607
        0.45786666666666664, 0.010000000000000009
        0.4917999999999999, 0.016666666666666607
        0.5237, 0.026666666666666616
        0.5456666666666666, 0.029999999999999805
        0.5615666666666667, 0.040000000000000036
        0.5894666666666667, 0.050000000000000044
        0.6073333333333334, 0.06333333333333324
        0.6252000000000001, 0.07666666666666666
        0.6410666666666667, 0.09000000000000008
        0.6569333333333334, 0.10333333333333328
        0.6668333333333334, 0.11333333333333329
        0.6806000000000001, 0.13666666666666671
        0.7336333333333334, 0.23333333333333328
        0.7437333333333334, 0.22333333333333327
        0.7766666666666666, 0.33000000000000007
        0.8199333333333332, 0.4033333333333333
        0.8115333333333334, 0.44333333333333336
        0.8366333333333331, 0.5333333333333334
        0.9662, 0.7766666666666668
        0.9748666666666665, 0.9100000000000003
        0.9871666666666667, 0.8800000000000002
        0.9821666666666666, 0.9800000000000002
        0.921, 0.6966666666666668
        0.9347666666666665, 0.7200000000000002
        0.8718333333333332, 0.6133333333333334"""

        kr_drain_nwp = """0.23499999999999988, 0.896666666666667
        0.2646333333333333, 0.9333333333333336
        0.2502, 0.9766666666666669
        0.2898666666666666, 0.8100000000000002
        0.3065, 0.7466666666666668
        0.3472333333333333, 0.6733333333333335
        0.3659, 0.6066666666666668
        0.3967333333333334, 0.5233333333333334
        0.4125, 0.5466666666666667
        0.43346666666666667, 0.45000000000000007
        0.5799333333333333, 0.20333333333333337
        0.5433333333333333, 0.2633333333333333
        0.4723666666666667, 0.36
        0.48016666666666663, 0.38
        0.6263666666666667, 0.15999999999999992
        0.6021000000000001, 0.18666666666666654
        0.6789333333333334, 0.10333333333333328
        0.6546333333333334, 0.1333333333333333
        0.7070666666666667, 0.09000000000000008
        0.7292333333333334, 0.07333333333333325
        0.7653666666666668, 0.05999999999999983
        0.7453333333333334, 0.06333333333333324
        0.7795666666666667, 0.040000000000000036
        0.7995666666666668, 0.040000000000000036
        0.8196000000000001, 0.036666666666666625
        0.8336999999999999, 0.026666666666666616
        0.8597000000000001, 0.026666666666666616
        0.8777999999999999, 0.016666666666666607
        0.8998666666666666, 0.010000000000000009
        0.9218333333333333, 0.01333333333333342
        0.9398333333333333, 0.01333333333333342
        0.9598666666666666, 0.010000000000000009
        0.9858666666666667, 0.010000000000000009
        1.0019333333333333, 0.0033333333333334103"""

        kr_imb_wp = """0.22970297029702974, 0.014062762208423507
        0.22772277227722776, 0.003859708004698703
        0.23960396039603954, 0.007450914582983081
        0.25148514851485154, 0.014431951669742649
        0.26138613861386134, 0.007820104044302223
        0.2752475247524753, 0.008055042792414202
        0.30495049504950505, 0.008558482966940506
        0.3326732673267328, 0.01241819097163921
        0.3623762376237623, 0.009531800637690324
        0.3861386138613861, 0.009934552777311367
        0.4099009900990099, 0.013727135425406711
        0.4415841584158416, 0.02104379929518352
        0.4277227722772278, 0.01402919953012205
        0.46732673267326735, 0.018090283604631274
        0.500990099009901, 0.025440510152709983
        0.5287128712871287, 0.029300218157408686
        0.5643564356435644, 0.04346366840073834
        0.5920792079207922, 0.05410303742238609
        0.6198019801980199, 0.07491189796945763
        0.6435643564356437, 0.08548414163450202
        0.6673267326732673, 0.10283604631649546
        0.691089108910891, 0.12018795099848933
        0.700990099009901, 0.1339150864238965
        0.7207920792079208, 0.15458969625776087
        0.7346534653465346, 0.17516361805672043
        0.7425742574257426, 0.19563685182077517"""

        kr_imb_nwp = """0.23960396039603954, 0.9159254908541694
        0.3188118811881188, 0.8664205403591199
        0.34851485148514855, 0.7584494042624592
        0.37227722772277233, 0.7215640208088601
        0.40198019801980206, 0.681389494881691
        0.4297029702970298, 0.6106729316999489
        0.28316831683168325, 0.9336130223191803
        0.24950495049504956, 0.9737204228897456
        0.4594059405940594, 0.5366001006880343
        0.4950495049504951, 0.4490686356771265
        0.5128712871287129, 0.3781842591038762
        0.5069306930693069, 0.3136767914079539
        0.5366336633663367, 0.2057056553112937
        0.5762376237623763, 0.23349555294512436
        0.5821782178217823, 0.18274878335291111
        0.6237623762376239, 0.15294512502097612
        0.6495049504950496, 0.11609330424567865
        0.6811881188118811, 0.0691726799798622
        0.697029702970297, 0.04571236784695376
        0.7188118811881188, 0.029132404765900066
        0.6554455445544554, 0.03483806007719403
        0.6792079207920794, 0.021681490182916097
        0.697029702970297, 0.018593723779157134
        0.7366336633663366, 0.02265480785366636
        0.7564356435643564, 0.0229904346366836
        0.5465346534653466, 0.3007887229400903"""

        kr_drain_wp   = np.loadtxt(kr_drain_wp.splitlines(), delimiter=',')
        kr_drain_nwp  = np.loadtxt(kr_drain_nwp.splitlines(), delimiter=',')
        kr_imb_wp     = np.loadtxt(kr_imb_wp.splitlines(), delimiter=',')
        kr_imb_nwp    = np.loadtxt(kr_imb_nwp.splitlines(), delimiter=',')

        

        return {
            "drainageSimulation": [kr_drain_wp, kr_drain_nwp],
            "imbibitionSimulation": [kr_imb_wp, kr_imb_nwp]
        }
