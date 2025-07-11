from utils.plots.plotter import Plotter2D, Plotter3D
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import count
import moviepy.video.io.ImageSequenceClip
import openpnm as op
from PIL import Image

class PostProcessing:
    def __init__(self, algorithm, base_path):
        self.algorithm = algorithm
        self.base_path = base_path
        self.graph_path = os.path.join(base_path, 'results', self.algorithm.network.project_name, 'graphs')
        self.video_path = os.path.join(base_path, 'results', self.algorithm.network.project_name, 'videos')
        self.frame_path = os.path.join(self.video_path, 'frames')
        
        os.makedirs(self.graph_path, exist_ok=True)
        os.makedirs(self.frame_path , exist_ok=True)
        
    def plot_network(self, lwidth=3, msize=100):
        pn = self.algorithm.network.network
        phases = self.algorithm.phases.phases
        dim = self.algorithm.network.dim
        linewidth = pn['throat.diameter'] / pn['throat.diameter'].max() * lwidth
        markersize = pn['pore.diameter'] / pn['pore.diameter'].max() * msize
        if dim == '3D':
            plotter = Plotter3D(layout='pore_network_3d')
        else:
            plotter = Plotter2D(layout='pore_network_2d')
        ax = plotter.ax
        first_phase = self.algorithm.algorithm[0].settings.phase
        phase_ic_color = next(p["color"] for p in phases if p["name"] != first_phase)
        op.visualization.plot_coordinates(pn, markersize=markersize, c=phase_ic_color, zorder=2, alpha=0.8, ax=ax)
        op.visualization.plot_connections(pn, linewidth=linewidth, c=phase_ic_color, zorder=1, alpha=0.8, ax=ax)
        plotter.apply_layout()
        plotter.save(os.path.join(self.graph_path, f'Network_{self.algorithm.network.project_name}.png'))
        return
    
    def make_invasion(self, lwidth=3, msize=100):
        pn = self.algorithm.network.network
        phases = self.algorithm.phases.phases
        dim = self.algorithm.network.dim
        linewidth = pn['throat.diameter'] / pn['throat.diameter'].max() * lwidth
        markersize = pn['pore.diameter'] / pn['pore.diameter'].max() * msize
        self.invasion_path = os.path.join(self.frame_path, 'invasion_frames')
        os.makedirs(self.invasion_path, exist_ok=True)
        _frame_id = count()
        centroids = pn.coords[pn.conns].mean(axis=1)
        x_throat, y_throat = centroids[:, 0], centroids[:, 1]
        for alg in self.algorithm.algorithm:
            inv_phase = alg.settings.phase
            inv_color = next(p['color'] for p in phases if p['name'] == inv_phase)
            not_inv_color = next(p['color'] for p in phases if p['name'] != inv_phase)
            entry_pressure = (alg.project[inv_phase]['throat.entry_pressure'])
            throats_ic = pn.Ts[alg['throat.ic_invaded']]
            pores_ic = pn.Ps[alg['pore.ic_invaded']]
            invasion_sequence = np.unique(
                alg['throat.invasion_sequence'][np.isfinite(alg['throat.invasion_sequence'])])
            for seq in invasion_sequence:
                p_kpa = alg['throat.invasion_pressure'][
                  alg['throat.invasion_sequence'] == seq].max() / 1000
                if dim == '3D':
                    plotter = Plotter3D(layout='invasion_3d', title=f'Pressure = {p_kpa:.2f} kPa')
                else:
                    plotter = Plotter2D(layout='invasion_2d', title=f'Pressure = {p_kpa:.2f} kPa')
                ax = plotter.ax
                self._draw_invasion(ax, pn, alg, seq, x_throat, y_throat, entry_pressure, inv_color, 
                                    not_inv_color, linewidth, markersize, throats_ic, pores_ic)
                idx = next(_frame_id)
                plotter.apply_layout()
                plotter.save(os.path.join(self.invasion_path, f'invasion_{idx:04d}.png'))
        return self.invasion_path
    
    def make_clusters(self, lwidth=3, msize=100):
        pn = self.algorithm.network.network
        dim = self.algorithm.network.dim
        linewidth = pn['throat.diameter'] / pn['throat.diameter'].max() * lwidth
        markersize = pn['pore.diameter'] / pn['pore.diameter'].max() * msize
        self.clusters_path = os.path.join(self.frame_path, 'clusters_frames')
        os.makedirs(self.clusters_path, exist_ok=True)
        _frame_id = count()
        for alg in self.algorithm.algorithm:
            invasion_sequence = np.unique(
                alg['throat.invasion_sequence'][np.isfinite(alg['throat.invasion_sequence'])])
            for seq in invasion_sequence:
                if dim == '3D':
                    plotter = Plotter3D(layout='invasion_3d', title='Clusters / Trapping')
                else:
                    plotter = Plotter2D(layout='invasion_2d', title='Clusters / Trapping')
                ax = plotter.ax
                self._draw_clusters(ax, pn, alg, seq,linewidth, markersize)
                idx = next(_frame_id)
                plotter.apply_layout()
                plotter.save(os.path.join(self.clusters_path, f'clusters_{idx:04d}.png'))
        return self.clusters_path
    
    def make_frames_side_by_side(self):
        if not hasattr(self, 'invasion_path'):
            self.make_invasion()
        if not hasattr(self, 'clusters_path'):
            self.make_clusters()
        invasion_path = self.invasion_path
        clusters_path = self.clusters_path
        
        invasion_files = sorted([f for f in os.listdir(invasion_path) if f.endswith('.png')])
        clusters_files = sorted([f for f in os.listdir(clusters_path) if f.endswith('.png')])
        self.frames_side_by_side = os.path.join(self.frame_path, 'frames_side_by_side')
        os.makedirs(self.frames_side_by_side, exist_ok=True)
        for inv_file, cl_file in zip(invasion_files, clusters_files):
            self.save_images_side_by_side(
                os.path.join(invasion_path, inv_file),
                os.path.join(clusters_path, cl_file),
                os.path.join(self.frames_side_by_side, f'side_by_side_{inv_file.split("_")[-1]}')
            )
        return self.frames_side_by_side
    
    def make_video(self, frames_path, fps=5, output_file=None):
        files = os.listdir(frames_path)
        files = [os.path.join(frames_path, file)  for file in files if os.path.isfile(os.path.join(frames_path, file)) and file.lower().endswith('.png')]
        files = sorted(files)
        if not files:
            raise RuntimeError('No frames found to make video.')
        files.insert(0, files[0])
        files.append(files[-1])
        output_file = output_file or os.path.join(self.video_path, 'video.mp4')
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(files, fps=fps)
        clip.write_videofile(output_file)
        return output_file
    
    def plot_relative_permeability(self, alg, Snwp_num=20, output_file=None):
        pn = self.algorithm.network.network
        wp = self.algorithm.phases.get_wetting_phase()
        wp_model = wp['model']
        nwp = self.algorithm.phases.get_non_wetting_phase()
        nwp_model = nwp['model']
        self.algorithm.phases.add_conduit_conductance_model(wp_model)
        self.algorithm.phases.add_conduit_conductance_model(nwp_model)
        inlet = pn.Ps[alg['pore.bc.inlet']]
        outlet = pn.Ps[alg['pore.bc.outlet']]
        
        def update_occupancy_and_get_saturation(network, nwp, wp, alg, seq):
            # Determine which phase is the invading phase
            inv_phase_name = alg.settings.phase
            non_wetting_phase = self.algorithm.phases.get_non_wetting_phase()
            is_invading_nwp = (inv_phase_name == non_wetting_phase['name'])

            # Find invaded pores/throats at this sequence
            invaded_pores = alg['pore.invasion_sequence'] < seq
            invaded_throats = alg['throat.invasion_sequence'] < seq

            # Set occupancy for both phases
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

        tmask = np.isfinite(alg['throat.invasion_sequence']) & (alg['throat.invasion_sequence'] > 0)
        max_seq = np.max(alg['throat.invasion_sequence'][tmask])
        min_seq = np.min(alg['throat.invasion_sequence'][tmask])
        relperm_sequence = np.linspace(min_seq, max_seq, Snwp_num).astype(int)
        Snwparr, relperm_nwp, relperm_wp = [], [], []

        for i in relperm_sequence:
            sat = update_occupancy_and_get_saturation(pn, nwp_model, wp_model, alg, i)
            Snwparr.append(sat*100)  # Convert to percentage
            Rate_abs_nwp = Rate_calc(pn, nwp_model, inlet, outlet, conductance='throat.hydraulic_conductance')
            Rate_abs_wp = Rate_calc(pn, wp_model, inlet, outlet, conductance='throat.hydraulic_conductance')
            Rate_enwp = Rate_calc(pn, nwp_model, inlet, outlet, conductance='throat.conduit_hydraulic_conductance')
            Rate_ewp = Rate_calc(pn, wp_model, inlet, outlet, conductance='throat.conduit_hydraulic_conductance')
            relperm_nwp.append(Rate_enwp / Rate_abs_nwp)
            relperm_wp.append(Rate_ewp / Rate_abs_wp)

        plotter = Plotter2D(layout='relative_permeability', title=f'RP {alg.name}')
        ax = plotter.ax
        ax.plot(Snwparr, relperm_nwp, '-o', label='Kr_nwp', color=nwp['color'])
        ax.plot(Snwparr, relperm_wp, '-*', label='Kr_wp', color=wp['color'])
        output_file = output_file or os.path.join(self.graph_path, f'RP_{alg.name}.png')
        plotter.apply_layout()
        plotter.save(output_file)
        return output_file

    def _draw_invasion(self, ax, pn, alg, sequence,
                      x_throat, y_throat, entry_pressure,
                      inv_color, not_inv_color,
                      linewidth, markersize,
                      throats_ic, pores_ic):
        # Plot the throat invasion pressure as text on the plot
        # for x, y, p in zip(x_throat, y_throat, entry_pressure):
        #     ax.text(x, y, f'{p:.2f}', fontsize=6,
        #             ha='center', va='center', color='black', zorder=3)
        
        #Plot inicial conditions with minor alpha
        self._plot_pores_and_throats(pn, pores=pores_ic, throats=throats_ic,
                          color=inv_color, alpha=0.5, 
                          markersize=markersize, linewidth=linewidth, ax=ax)          
          
        mask_throat = alg['throat.invasion_sequence'] <= sequence
        mask_pore   = alg['pore.invasion_sequence']   <= sequence
        new_throats = np.setdiff1d(pn.Ts[mask_throat], throats_ic)
        new_pores   = np.setdiff1d(pn.Ps[mask_pore], pores_ic)
        still_not_t = np.setdiff1d(pn.Ts[~mask_throat], throats_ic)
        still_not_p = np.setdiff1d( pn.Ps[~mask_pore], pores_ic)
        
        # Plot the new throats and pores
        self._plot_pores_and_throats(pn, pores=new_pores, throats=new_throats,
                          color=inv_color, alpha=0.8, 
                          markersize=markersize, linewidth=linewidth, ax=ax)
            
        # Plot the still not invaded throats and pores
        self._plot_pores_and_throats(pn, pores=still_not_p, throats=still_not_t,
                          color=not_inv_color, alpha=0.8, 
                          markersize=markersize, linewidth=linewidth, ax=ax)
    
    def _draw_clusters(self, ax, pn, alg, sequence,linewidth, markersize):
        p = alg['throat.invasion_pressure'][
            alg['throat.invasion_sequence'] == sequence].max()
        pseq = alg['pore.invasion_pressure']
        tseq = alg['throat.invasion_pressure']
        occupied = pseq > p
        s, b = op._skgraph.simulations.site_percolation(
            conns=pn.conns, occupied_sites=occupied)
        # Identify uninvaded throats between previously invaded pores within same cluster   
        both_pores_invaded = (pseq[alg.network.conns[:, 0]] <= p) & (pseq[alg.network.conns[:, 1]] <= p)
        same_cluster = s[alg.network.conns[:, 0]] == s[alg.network.conns[:, 1]]
        uninvaded_throat = tseq > p
        trap_condition = both_pores_invaded & same_cluster & uninvaded_throat
        trapped_throats = trap_condition
        
        clusters_out = np.unique(s[alg['pore.bc.outlet']])
        Ts = pn.find_neighbor_throats(pores=s >= 0)
        b[Ts] = np.amax(s[pn.conns], axis=1)[Ts]
        trapped_pores   = np.isin(s, clusters_out, invert=True) & (s >= 0)
        trapped_throats += np.isin(b, clusters_out, invert=True) & (b >= 0)

        self._plot_pores_and_throats(pn, pores=trapped_pores, linewidth=linewidth,markersize=markersize, 
                                     color_by=s[trapped_pores], ax=ax)
        self._plot_pores_and_throats(pn, throats=trapped_throats, linewidth=linewidth,markersize=markersize,  
                                     color_by=b[trapped_throats], ax=ax)
        mask_inv_p = pseq <= p
        mask_inv_t = alg['throat.invasion_pressure'] <= p
        self._plot_pores_and_throats(pn, pores=mask_inv_p, linewidth=linewidth,markersize=markersize, 
                                     c='k', ax=ax)
        self._plot_pores_and_throats(pn, throats=mask_inv_t, linewidth=linewidth,markersize=markersize, 
                                     c='k', linestyle='--', ax=ax)
        #draw empty pores
        self._plot_pores_and_throats(pn, pores=pn.Ps,linewidth=linewidth,markersize=markersize,
                                     color='k', alpha=0.0, ax=ax)
            
    def _plot_pores_and_throats(self, pn, pores=None, throats=None, markersize=None, 
                                linewidth=None, ax=None, **kwargs):
        
        if throats is not None:
            if throats.any():
                op.visualization.plot_connections(
                pn, throats,zorder=1, ax=ax, **kwargs,
                linewidth=linewidth[throats] if linewidth is not None else None,
                )
            
        if pores is not None:
            if pores.any():
                op.visualization.plot_coordinates(
                pn, pores, zorder=2, ax=ax,  **kwargs,
                markersize=markersize[pores] if markersize is not None else None,
                )
                
    def save_images_side_by_side(self, file1, file2, outfile):
        img1 = Image.open(file1)
        img2 = Image.open(file2)
        total_width = img1.width + img2.width
        max_height = max(img1.height, img2.height)
        new_img = Image.new('RGB', (total_width, max_height))
        new_img.paste(img1, (0, 0))
        new_img.paste(img2, (img1.width, 0))
        new_img.save(outfile)