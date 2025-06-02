# Storage and Network Data and Topology
## importing Packages
import os
import openpnm as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


op.visualization.set_mpl_style()
path = os.path.dirname(__file__)


msize = 100
lwidth = 5
np.random.seed(0)

# Ramdom network 
pn = op.network.Demo(shape=[4, 1, 1])


# Pandas DataFrame of Network Data
pore_data_sheet = pd.DataFrame({k: pn[k] for k in pn.props(element='pore') if pn[k].ndim == 1})

print(pore_data_sheet.head())

column = pore_data_sheet['pore.volume']
print(column)

pn['throat.list'] = [1, 2, 3]
print(type(pn['throat.list']))


pn['pore.label'] = False
print(pn.labels(element='pore'))
print(pn.pores('label'))

pn['pore.label'][[0, 1, 2]] = True
print(pn.pores('label'))

pn.set_label('another', pores=[2, 3])
print(pn.pores('another'))

pn['pore._hidden'] = 1
print(pn.props())


# Rerpresenting Topology
np.random.seed(0)

fig1 = plt.figure(figsize=[6,5]) 
# dn = op.network.Delaunay(points=30, shape=[5, 5, 0])
pn = op.network.Cubic(shape=[5, 5, 1], spacing=1)
ax1 = op.visualization.plot_tutorial(pn)
fig1.savefig(path+"/demo/Topology.png")


# Conduite Data: Pore - Throat - Pore
# print("###############",pn.get_conduit_data())
print(pn)
pn['pore.diameter'] = np.random.rand(pn.Np)
pn['throat.diameter'] = np.random.rand(pn.Nt)
D = pn.get_conduit_data('diameter')
print("D=",D)

print(pn.conns)

R1_R2 = pn['pore.diameter'][pn.conns]
print("Pore_diameters=",R1_R2)


# Adjacency Matrix shows which pores are connected to which pores.
# It is normally sparce, so the todence() method can be used.
am = pn.create_adjacency_matrix().todense()
print("AM =",am)

# Incidence Matrix shows which pores(rows) are connected to which throats(columns).
# It is normally sparce, so the todence() method can be used.
im = pn.create_incidence_matrix().todense()
print("IM=",im)

# Finding Neighbors
P_left = pn.pores('left')
P_right = pn.pores('right')
print(P_left)
print(P_right)

InletPores = pn.pores(['left'])
print(InletPores)
InletNeighborPores = pn.find_neighbor_pores(pores=InletPores, mode='or') # mode='xnor'
print(InletNeighborPores)

P_all = pn.pores()
T_all = pn.throats()

fig2, ax2 = plt.subplots()
op.visualization.plot_coordinates(pn, pn.Ps, c='lightgrey', 
                                  size_by=pn['pore.diameter'],markersize=msize, ax=ax2,zorder=1)
op.visualization.plot_connections(pn, T_all, ax=ax2,
                                  size_by=pn['throat.diameter'],linewidth=lwidth, c='lightgrey',zorder=2)
op.visualization.plot_coordinates(pn, P_left, c='red', marker='*', 
                                  size_by=pn['pore.diameter'],markersize=msize, ax=ax2,zorder=3)
op.visualization.plot_coordinates(pn, P_right, c='blue', marker='.', 
                                  size_by=pn['pore.diameter'],markersize=msize, ax=ax2,zorder=3)
fig2.savefig(path+"/demo_results/InletOutlet.png")



fig3, ax3 = plt.subplots()
op.visualization.plot_coordinates(pn, P_all, c='lightgrey',
                                  size_by=pn['pore.diameter'],markersize=msize, ax=ax3,zorder=0)
op.visualization.plot_connections(pn, T_all, ax=ax3,
                                  size_by=pn['throat.diameter'],linewidth=lwidth, c='lightgrey',zorder=1)
op.visualization.plot_coordinates(pn, P_left, c='red', 
                                  size_by=pn['pore.diameter'],markersize=msize, marker='*', ax=ax3,zorder=2)
op.visualization.plot_coordinates(pn, P_right, c='blue', 
                                  size_by=pn['pore.diameter'],markersize=msize, marker='.', ax=ax3,zorder=2)
op.visualization.plot_coordinates(pn, InletNeighborPores, c='green', 
                                  size_by=pn['pore.diameter'],markersize=msize, marker='s', ax=ax3,zorder=2);
fig3.savefig(path+"/demo_results/NeighborPores.png")


InletNeighborThroats = pn.find_neighbor_throats(pores=InletPores, mode='or') # mode='xnor'

fig4, ax4 = plt.subplots()
op.visualization.plot_connections(pn, T_all, ax=ax4,
                                  size_by=pn['throat.diameter'],linewidth=lwidth, c='lightgrey',zorder=0)
op.visualization.plot_connections(pn, InletNeighborThroats, ax=ax4,
                                  size_by=pn['throat.diameter'],linewidth=lwidth,zorder=1)
op.visualization.plot_coordinates(pn, P_all, c='lightgrey', 
                                  size_by=pn['pore.diameter'],markersize=msize, ax=ax4,zorder=2)
op.visualization.plot_coordinates(pn, P_left, c='red', 
                                  size_by=pn['pore.diameter'],markersize=msize, marker='*', ax=ax4,zorder=2)
op.visualization.plot_coordinates(pn, P_right, c='blue', 
                                  size_by=pn['pore.diameter'],markersize=msize, marker='.', ax=ax4,zorder=2);
fig4.savefig(path+"/demo_results/NeighborThroats.png")
