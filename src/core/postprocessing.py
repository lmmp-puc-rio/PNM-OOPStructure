import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import count
import moviepy.video.io.ImageSequenceClip
import openpnm as op

class PostProcessing:
    def __init__(self, algorithm, base_path):
        self.algorithm = algorithm
        self.base_path = base_path
        self.graph_path = os.path.join(base_path, 'results', self.algorithm.network.project_name, 'graphs')
        self.video_path = os.path.join(base_path, 'results', self.algorithm.network.project_name, 'videos')
        self.frame_path = os.path.join(self.video_path, 'frames')
        
        os.makedirs(self.graph_path, exist_ok=True)
        os.makedirs(self.frame_path , exist_ok=True)
        
    def plot_network(self, lwidth=3, msize=100, azim=-60, elev=15):
        pn = self.algorithm.network.network
        phases = self.algorithm.phases.phases
        dim =  self.algorithm.network.dim
        linewidth = pn['throat.diameter'] / pn['throat.diameter'].max() * lwidth + (lwidth/3)
        markersize = pn['pore.diameter'] / pn['pore.diameter'].max() * msize + msize
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(111, projection = '3d') if dim == '3D' else fig0.add_subplot(111)
        fig0.set_size_inches(10,10)
        ax0.set_aspect('auto')
        ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        ax0.grid(False)
        ax0.set_title(f'Pore Network',fontsize=16)
        if dim == '3D':
            ax0.view_init(elev=elev, azim=azim)
        first_phase = self.algorithm.algorithm[0].settings.phase
        phase_ic_color = next(p["color"] for p in phases if p["name"] != first_phase)
        op.visualization.plot_coordinates(pn, markersize=markersize, c=phase_ic_color,zorder=2 ,alpha=0.8, ax=ax0)
        op.visualization.plot_connections(pn, linewidth=linewidth, c=phase_ic_color,zorder=1 ,alpha=0.8, ax=ax0)
        fig0.savefig(os.path.join(self.graph_path, f'Network_{self.algorithm.network.project_name}.png'))
        plt.close(fig0)
        return
    
    def make_invasion(self, lwidth=3, msize=100):
        pn = self.algorithm.network.network
        phases = self.algorithm.phases
        linewidth = pn['throat.diameter'] / pn['throat.diameter'].max() * lwidth + (lwidth/3)
        markersize = pn['pore.diameter'] / pn['pore.diameter'].max() * msize + msize
        invasion_path = os.path.join(self.frame_path, 'invasion_frames')
        os.makedirs(invasion_path, exist_ok=True)
        _frame_id = count()
        centroids = pn.coords[pn.conns].mean(axis=1)
        x_throat, y_throat = centroids[:, 0], centroids[:, 1]
        for alg in self.algorithm.algorithm:
            inv_phase = alg.settings.phase
            inv_color     = next(p['color'] for p in phases.phases if p['name'] == inv_phase)
            not_inv_color = next(p['color'] for p in phases.phases if p['name'] != inv_phase)
            entry_pressure = (alg.project[inv_phase]['throat.entry_pressure'])
            throats_ic = pn.Ts[alg['throat.ic_invaded']]
            pores_ic   = pn.Ps[alg['pore.ic_invaded']]
            invasion_sequence = np.unique(
                alg['throat.invasion_sequence'][np.isfinite(alg['throat.invasion_sequence'])])
            fig, ax = plt.subplots(figsize=(6, 6))
            
            for seq in invasion_sequence:
                self._draw_invasion(ax, pn, alg, seq, x_throat, y_throat, entry_pressure, inv_color, not_inv_color, linewidth, markersize, throats_ic, pores_ic)
                fig.tight_layout()
                idx = next(_frame_id)
                fig.savefig(os.path.join(invasion_path, f'invasion_{idx:04d}.png'), dpi=150)
        return invasion_path
    
    def _draw_invasion(self, ax, pn, alg, sequence,
                      x_throat, y_throat, entry_pressure,
                      inv_color, not_inv_color,
                      linewidth, markersize,
                      throats_ic, pores_ic):
        plt.sca(ax)
        self._clear_ax(ax)
        # Plot the throat invasion pressure as text on the plot
        # for x, y, p in zip(x_throat, y_throat, entry_pressure):
        #     ax.text(x, y, f'{p:.2f}', fontsize=6,
        #             ha='center', va='center', color='black', zorder=3)
        if throats_ic.size:
            op.visualization.plot_connections(
                pn, throats_ic, alpha=0.5,
                linewidth=linewidth[throats_ic], c=inv_color, zorder=1, ax=ax)
        if pores_ic.size:
            op.visualization.plot_coordinates(
                pn, pores_ic, alpha=0.5,
                markersize=markersize[pores_ic], c=inv_color,zorder=2, ax=ax)
        throats_not_ic = np.setdiff1d(pn.Ts, throats_ic)
        pores_not_ic   = np.setdiff1d(pn.Ps, pores_ic)
        if throats_not_ic.size:
            op.visualization.plot_connections(
                pn, throats_not_ic, alpha=0.8,
                linewidth=linewidth[throats_not_ic], c=not_inv_color, zorder=1, ax=ax)
        if pores_not_ic.size:
            op.visualization.plot_coordinates(
                pn, pores_not_ic, alpha=0.8,
                markersize=markersize[pores_not_ic], c=not_inv_color,zorder=2, ax=ax)
        mask_throat = alg['throat.invasion_sequence'] <= sequence
        mask_pore   = alg['pore.invasion_sequence']   <= sequence
        new_throats = pn.Ts[mask_throat]
        new_pores   = pn.Ps[mask_pore]
        still_not_t = np.setdiff1d(throats_not_ic, new_throats)
        still_not_p = np.setdiff1d(pores_not_ic,   new_pores)
        if new_throats.size:
            op.visualization.plot_connections(
                pn, new_throats, alpha=0.8,
                linewidth=linewidth[new_throats], c=inv_color, zorder=1, ax=ax)
        if still_not_t.size:
            op.visualization.plot_connections(
                pn, still_not_t, alpha=0.8,
                linewidth=linewidth[still_not_t], c=not_inv_color, zorder=1, ax=ax)
        if new_pores.size:
            op.visualization.plot_coordinates(
                pn, new_pores, alpha=0.8,
                markersize=markersize[new_pores], c=inv_color,zorder=2, ax=ax)
        if still_not_p.size:
            op.visualization.plot_coordinates(
                pn, still_not_p, alpha=0.8,
                markersize=markersize[still_not_p], c=not_inv_color,zorder=2, ax=ax)
        p_kpa = alg['throat.invasion_pressure'][
                  alg['throat.invasion_sequence'] == sequence].max() / 1000
        ax.set_title(f'Pressure = {p_kpa:.2f} kPa', fontsize=10)
        ax.axis('off')
        
    def _clear_ax(self, ax):
        for art in ax.lines[:] + ax.collections[:]:
            art.remove()
        for txt in ax.texts[:]:
            txt.remove()
        leg = ax.get_legend()
        if leg:
            leg.remove()