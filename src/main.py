from utils.config_parser import ConfigParser
from core.network import Network
from core.phases import Phases
from core.algorithm import Algorithm
import os as os
import matplotlib.pyplot as plt
import numpy as np
import openpnm as op

path = os.path.dirname(__file__)
json_file = 'data/base.json'

cfg = ConfigParser.from_file(json_file)

pn = Network(config = cfg)
phases = Phases(network = pn, config = cfg)
algorithm = Algorithm(network = pn, phases = phases,config = cfg)
algorithm.run()

#inputs
lwidth = 8
msize = 200
azim = -60
elev = 15


#setting throats and pores plot diameters
linewidth = pn.network['throat.diameter'] / pn.network['throat.diameter'].max() * lwidth
markersize = pn.network['pore.diameter'] / pn.network['pore.diameter'].max() * msize

Np_col = len(np.unique(pn.network['pore.coords'].T[0]))
Np_row = len(np.unique(pn.network['pore.coords'].T[1]))
pn_dim = '2D'
if len(np.unique(pn.network['pore.coords'].T[2])) > 1:
    pn_dim = '3D'
graph_path = os.path.join(path, 'results', f'DRRelPerm{pn_dim}', 'graphs')
video_path = os.path.join(path, 'results', f'DRRelPerm{pn_dim}', 'videos')
frame_path = os.path.join(video_path, f'frames_{Np_col}_pores')
os.makedirs(graph_path, exist_ok=True)
os.makedirs(frame_path , exist_ok=True)



#Network figure
fig0 = plt.figure()
ax0 = fig0.add_subplot(111, projection = '3d') if pn_dim == '3D' else fig0.add_subplot(111)
fig0.set_size_inches(Np_col,Np_row)
ax0.set_aspect('auto')
ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
ax0.grid(False)
ax0.set_title(f'Pore Network',fontsize=16)
if pn_dim == '3D':
    ax0.view_init(elev=elev, azim=azim)
    

first_phase = algorithm.algorithm[0].settings.phase
phase_ic_color = next(p["color"] for p in phases.phases if p["name"] != first_phase)
op.visualization.plot_coordinates(pn.network, markersize=markersize, c=phase_ic_color, alpha=0.8, ax=ax0)
op.visualization.plot_connections(pn.network, linewidth=linewidth, c=phase_ic_color, alpha=0.8, ax=ax0)
fig0.savefig(os.path.join(graph_path, f'Network{pn_dim}_{Np_col}.png'))


centroids = pn.network.coords[pn.network.conns].mean(axis=1)
x_throat = centroids[:, 0] 
y_throat = centroids[:, 1]


def clear_ax(ax):
    for collection in ax.collections:
            collection.remove()
            
def clear_text_ax(ax):
    for text in ax.texts:
            text.remove()

k = 0

#algorithm figure
def algorithm_figure(alg,fig,ax):
    global k
    
    clear_ax(ax)
    inv_phase = alg.settings.phase
    phase_model = alg.project[inv_phase]
    entry_pressure = phase_model['throat.entry_pressure']
    entry_pressure = entry_pressure/1000
    for x, y, p in zip(x_throat, y_throat, entry_pressure):
        ax.text(x, y,
                f'{p:.2f}',
                fontsize=8,
                ha='center', va='center',
                color='black', 
                zorder=3)
        
    
    inv_color = next(p["color"] for p in phases.phases if p["name"] == inv_phase)
    not_inv_color = next(p["color"] for p in phases.phases if p["name"] != inv_phase)
    throats_invaded_ic = pn.network.Ts[alg['throat.ic_invaded']].copy()
    pores_invaded_ic = pn.network.Ps[alg['pore.ic_invaded']].copy()
    throats_not_invaded_ic = np.setdiff1d(pn.network.Ts, throats_invaded_ic).astype(int)
    pores_not_invaded_ic = np.setdiff1d(pn.network.Ps, pores_invaded_ic).astype(int)

    if len(throats_invaded_ic)>0:
            op.visualization.plot_connections(pn.network, throats_invaded_ic, alpha=0.5, linewidth=linewidth[throats_invaded_ic], c=inv_color ,ax=ax)
    if len(pores_invaded_ic)>0:
        op.visualization.plot_coordinates(pn.network, pores_invaded_ic, alpha=0.5, markersize=markersize[pores_invaded_ic], c=inv_color,ax=ax)
    if len(throats_not_invaded_ic)>0:
            op.visualization.plot_connections(pn.network, throats_not_invaded_ic, alpha=0.8, linewidth=linewidth[throats_not_invaded_ic], c=not_inv_color ,ax=ax)
    if len(pores_not_invaded_ic)>0:
        op.visualization.plot_coordinates(pn.network, pores_not_invaded_ic, alpha=0.8, markersize=markersize[pores_not_invaded_ic], c=not_inv_color,ax=ax)
    
    ax.set_title(f'Initial Condition',fontsize=16)
    fig.savefig(os.path.join(frame_path,f'frame{k}.png'))
    k +=1
    
    invasion_sequence = np.unique(alg['throat.invasion_sequence'][np.isfinite(alg['throat.invasion_sequence'])])
    
    for sequence in invasion_sequence:
        clear_ax(ax)
        
        if len(throats_invaded_ic)>0:
            op.visualization.plot_connections(pn.network, throats_invaded_ic, alpha=0.5, linewidth=linewidth[throats_invaded_ic], c=inv_color ,ax=ax)
        if len(pores_invaded_ic)>0:
            op.visualization.plot_coordinates(pn.network, pores_invaded_ic, alpha=0.5, markersize=markersize[pores_invaded_ic], c=inv_color,ax=ax)
        invasion_pressure = max(alg['throat.invasion_pressure'][alg['throat.invasion_sequence'] == sequence])/1000
        inv_throat_pattern = alg['throat.invasion_sequence'] <= sequence
        inv_pore_pattern = alg['pore.invasion_sequence'] <= sequence
        
        # new_throats = np.setdiff1d(pn.network.Ts[inv_throat_pattern], throats_invaded_ic).astype(int)
        # new_pores = np.setdiff1d(pn.network.Ps[inv_pore_pattern], pores_invaded_ic).astype(int)
        
        new_throats = pn.network.Ts[inv_throat_pattern]
        new_pores =pn.network.Ps[inv_pore_pattern]
        
        
        throats_not_invaded = np.setdiff1d(throats_not_invaded_ic,new_throats)
        pores_not_invaded = np.setdiff1d(pores_not_invaded_ic,new_pores)
        
        if len(new_throats)>0:
            op.visualization.plot_connections(pn.network, new_throats, alpha=0.8, linewidth=linewidth[new_throats], c=inv_color ,ax=ax)
        if len(throats_not_invaded)>0:
            op.visualization.plot_connections(pn.network, throats_not_invaded, alpha=0.8, linewidth=linewidth[throats_not_invaded], c=not_inv_color ,ax=ax)
        
        if len(new_pores)>0:
            op.visualization.plot_coordinates(pn.network, new_pores, alpha=0.8, markersize=markersize[new_pores], c=inv_color,ax=ax)
        if len(pores_not_invaded)>0:
            op.visualization.plot_coordinates(pn.network, pores_not_invaded, alpha=0.8, markersize=markersize[pores_not_invaded], c=not_inv_color,ax=ax)
        
        ax.set_title(f'Pressure = {invasion_pressure:.2f} kPa',fontsize=16)
        fig.savefig(os.path.join(frame_path,f'frame{k}.png'))
        k +=1
    return

for alg in algorithm.algorithm:
    algorithm_figure(alg,fig0,ax0)
    clear_text_ax(ax0)
