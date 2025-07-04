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
        pn = self.algorithm.network
        phases = self.algorithm.phases
        algorithm = self.algorithm
        linewidth = pn.network['throat.diameter'] / pn.network['throat.diameter'].max() * lwidth + (lwidth/3)
        markersize = pn.network['pore.diameter'] / pn.network['pore.diameter'].max() * msize + msize
        Np_col = len(np.unique(pn.network.coords.T[0]))
        Np_row = len(np.unique(pn.network.coords.T[1]))
        pn_dim = '2D'
        if len(np.unique(pn.network['pore.coords'].T[2])) > 1:
            pn_dim = '3D'
        fig0 = plt.figure()
        ax0 = fig0.add_subplot(111, projection = '3d') if pn_dim == '3D' else fig0.add_subplot(111)
        fig0.set_size_inches(Np_col,Np_row)
        ax0.set_aspect('auto')
        ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        ax0.grid(False)
        ax0.set_title(f'Pore Network',fontsize=16)
        if pn_dim == '3D':
            ax0.view_init(elev=elev, azim=azim)
        first_phase = algorithm.algorithm[0].settings.phase
        phase_ic_color = next(p["color"] for p in phases.phases if p["name"] != first_phase)
        op.visualization.plot_coordinates(pn.network, markersize=markersize, c=phase_ic_color, alpha=0.8, ax=ax0)
        op.visualization.plot_connections(pn.network, linewidth=linewidth, c=phase_ic_color, alpha=0.8, ax=ax0)
        fig0.savefig(os.path.join(self.graph_path, f'Network_{self.algorithm.network.project_name}.png'))
        plt.close(fig0)
        return