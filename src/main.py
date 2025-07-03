from utils.config_parser import ConfigParser
from core.network import Network
from core.phases import Phases
from core.algorithm import Algorithm
import os as os
import matplotlib.pyplot as plt
import numpy as np
import openpnm as op
from itertools import count
from pathlib import Path

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

# ---------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------
_frame_id = count()            # 0, 1, 2, ...

def _clear_ax(ax):
    for art in ax.lines[:] + ax.collections[:]:
        art.remove()
    for txt in ax.texts[:]:
        txt.remove()
    leg = ax.get_legend()
    if leg:
        leg.remove()

# ---------------------------------------------------------------
# painel ESQUERDO  –––  invasão passo-a-passo
# ---------------------------------------------------------------
def _draw_invasion(ax, *, pn, alg, sequence,
                   x_throat, y_throat, entry_pressure,
                   inv_color, not_inv_color,
                   linewidth, markersize,
                   throats_ic, pores_ic):
    """
    Estado da invasão ATÉ 'sequence' (mesma aparência do seu algorithm_figure).
    """
    plt.sca(ax)  
    _clear_ax(ax)

    # --- 1. rótulo da entry-pressure em cada garganta -----------------
    for x, y, p in zip(x_throat, y_throat, entry_pressure):
        ax.text(x, y, f'{p:.2f}', fontsize=6,
                ha='center', va='center', color='black', zorder=3)

    # --- 2. condição inicial -----------------------------------------
    if throats_ic.size:
        op.visualization.plot_connections(
            pn.network, throats_ic, alpha=0.5,
            linewidth=linewidth[throats_ic], c=inv_color, ax=ax)
    if pores_ic.size:
        op.visualization.plot_coordinates(
            pn.network, pores_ic, alpha=0.5,
            markersize=markersize[pores_ic], c=inv_color, ax=ax)

    throats_not_ic = np.setdiff1d(pn.network.Ts, throats_ic)
    pores_not_ic   = np.setdiff1d(pn.network.Ps, pores_ic)

    if throats_not_ic.size:
        op.visualization.plot_connections(
            pn.network, throats_not_ic, alpha=0.8,
            linewidth=linewidth[throats_not_ic], c=not_inv_color, ax=ax)
    if pores_not_ic.size:
        op.visualization.plot_coordinates(
            pn.network, pores_not_ic, alpha=0.8,
            markersize=markersize[pores_not_ic], c=not_inv_color, ax=ax)

    # --- 3. elementos já invadidos até 'sequence' ---------------------
    mask_throat = alg['throat.invasion_sequence'] <= sequence
    mask_pore   = alg['pore.invasion_sequence']   <= sequence

    new_throats = pn.network.Ts[mask_throat]
    new_pores   = pn.network.Ps[mask_pore]

    still_not_t = np.setdiff1d(throats_not_ic, new_throats)
    still_not_p = np.setdiff1d(pores_not_ic,   new_pores)

    if new_throats.size:
        op.visualization.plot_connections(
            pn.network, new_throats, alpha=0.8,
            linewidth=linewidth[new_throats], c=inv_color, ax=ax)
    if still_not_t.size:
        op.visualization.plot_connections(
            pn.network, still_not_t, alpha=0.8,
            linewidth=linewidth[still_not_t], c=not_inv_color, ax=ax)

    if new_pores.size:
        op.visualization.plot_coordinates(
            pn.network, new_pores, alpha=0.8,
            markersize=markersize[new_pores], c=inv_color, ax=ax)
    if still_not_p.size:
        op.visualization.plot_coordinates(
            pn.network, still_not_p, alpha=0.8,
            markersize=markersize[still_not_p], c=not_inv_color, ax=ax)

    # --- 4. título ----------------------------------------------------
    p_kpa = alg['throat.invasion_pressure'][
              alg['throat.invasion_sequence'] == sequence].max() / 1_000
    ax.set_title(f'Pressure = {p_kpa:.2f} kPa', fontsize=10)
    ax.axis('off')

# ---------------------------------------------------------------
# painel DIREITO  –––  clusters / trapping  (sem mudanças)
# ---------------------------------------------------------------
def _draw_clusters(ax, pn, alg, sequence):
    plt.sca(ax)  
    _clear_ax(ax)

    p = alg['throat.invasion_pressure'][
        alg['throat.invasion_sequence'] == sequence].max()

    pseq     = alg['pore.invasion_pressure']
    occupied = pseq > p
    s, b = op._skgraph.simulations.site_percolation(
        conns=pn.conns, occupied_sites=occupied)

    clusters_out = np.unique(s[alg['pore.bc.outlet']])
    Ts = pn.find_neighbor_throats(pores=s >= 0)
    b[Ts] = np.amax(s[pn.conns], axis=1)[Ts]

    trapped_pores   = np.isin(s, clusters_out, invert=True) & (s >= 0)
    trapped_throats = np.isin(b, clusters_out, invert=True) & (b >= 0)

    if trapped_pores.any():
        op.visualization.plot_coordinates(
            pn, pores=trapped_pores, color_by=s[trapped_pores], ax=ax)

    mask_inv_p = pseq <= p
    if mask_inv_p.any():
        op.visualization.plot_coordinates(
            pn, pores=mask_inv_p, c='k', ax=ax)

    if trapped_throats.any():
        op.visualization.plot_connections(
            pn, throats=trapped_throats,
            color_by=b[trapped_throats], ax=ax)

    mask_inv_t = alg['throat.invasion_pressure'] <= p
    if mask_inv_t.any():
        op.visualization.plot_connections(
            pn, throats=mask_inv_t, c='k', linestyle='--', ax=ax)

    ax.set_title('Clusters / Trapping', fontsize=10)
    ax.axis('off')

# ---------------------------------------------------------------
# FUNÇÃO PRINCIPAL  –––  gera todos os frames
# ---------------------------------------------------------------
def make_frames(*, alg, pn, phases, frame_path):
    """
    Cria frame0000.png, frame0001.png, ... (invasão × clusters por pressure).
    """
    frame_path = Path(frame_path)
    frame_path.mkdir(parents=True, exist_ok=True)

    # ---------- dados globais que não mudam de frame ----------
    inv_phase = alg.settings.phase
    inv_color     = next(p['color'] for p in phases if p['name'] == inv_phase)
    not_inv_color = next(p['color'] for p in phases if p['name'] != inv_phase)

    # centróides de garganta p/ texto
    centroids = pn.network.coords[pn.network.conns].mean(axis=1)
    x_throat, y_throat = centroids[:, 0], centroids[:, 1]
    entry_pressure = (alg.project[inv_phase]['throat.entry_pressure']) / 1000  # kPa

    # condição inicial
    throats_ic = pn.network.Ts[alg['throat.ic_invaded']]
    pores_ic   = pn.network.Ps[alg['pore.ic_invaded']]

    # larguras / tamanhos default se não existirem no Network
    linewidth = pn.network['throat.diameter'] / pn.network['throat.diameter'].max() * 8
    markersize = pn.network['pore.diameter'] / pn.network['pore.diameter'].max() * 200

    invasion_sequence = np.unique(
        alg['throat.invasion_sequence'][np.isfinite(alg['throat.invasion_sequence'])])

    # ---------- gera um arquivo PNG por sequence ----------
    for seq in invasion_sequence:
        fig, (ax_L, ax_R) = plt.subplots(
            1, 2, figsize=(10, 5), subplot_kw={'aspect': 'equal'})

        _draw_invasion(
            ax_L, pn=pn, alg=alg, sequence=seq,
            x_throat=x_throat, y_throat=y_throat,
            entry_pressure=entry_pressure,
            inv_color=inv_color, not_inv_color=not_inv_color,
            linewidth=linewidth, markersize=markersize,
            throats_ic=throats_ic, pores_ic=pores_ic)

        _draw_clusters(ax_R, pn.network, alg, seq)

        fig.tight_layout()
        idx = next(_frame_id)
        fig.savefig(frame_path / f'frame{idx:04d}.png', dpi=150)
        plt.close(fig)


make_frames(
    alg        = algorithm.algorithm[0],        # objeto Drainage já executado
    pn         = pn,                  # rede de poros
    phases     = phases.phases,       # [{'name': 'water', 'color': '#0000ff'}, ...]
    frame_path = frame_path
)

make_frames(
    alg        = algorithm.algorithm[1],        # objeto Drainage já executado
    pn         = pn,                  # rede de poros
    phases     = phases.phases,       # [{'name': 'water', 'color': '#0000ff'}, ...]
    frame_path = frame_path
)