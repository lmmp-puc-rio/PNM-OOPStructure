# Cubic Network with Invasion Percolation
## importing Packages
import os, glob
import openpnm as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq as hq
import io
from PIL import Image, ImageGrab
import moviepy.video.io.ImageSequenceClip



op.visualization.set_mpl_style()
path = os.path.dirname(__file__)

# Create network and phases
"""First, we create a network and assign phases and properties in a similar way that we used to do for the other examples.""" 

# Creating a Cubic Network
Nx, Ny, Nz = 5, 5, 1
np.random.seed(5)
fps = 10
Lc = 1e-6
pn = op.network.Cubic(shape=[Nx, Ny, Nz], spacing=1e-4)
pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
pn.regenerate_models()
pn['pore.diameter'] = np.random.rand(pn.Np)*Lc
pn['throat.diameter'] = np.random.rand(pn.Nt)*Lc
air = op.phase.Air(network=pn)
water = op.phase.Water(network=pn,name='water')
water.add_model_collection(op.models.collections.phase.water)
water.add_model_collection(op.models.collections.physics.basic)
water.regenerate_models()


# Pore size distribution
air['pore.contact_angle'] = 120
air['pore.surface_tension'] = 0.072
f = op.models.physics.capillary_pressure.washburn
air.add_model(propname='throat.entry_pressure',
              model=f, 
              surface_tension='throat.surface_tension',
              contact_angle='throat.contact_angle',
              diameter='throat.diameter',)

# Network Visualization
fig1,ax1 = plt.subplots()
op.visualization.plot_coordinates(pn, pn.pores('left'), size_by=pn['pore.diameter'], markersize=50, c='r',ax=ax1)
op.visualization.plot_connections(pn, pn.throats() ,size_by=pn['throat.diameter'], linewidth=5, alpha=0.5 , c='b' ,ax=ax1)
op.visualization.plot_coordinates(pn, pn.pores('left',mode= 'nor'),size_by=pn['pore.diameter'], markersize=50, c='b', ax=ax1)
fig1.savefig(path+"/results/videos/ip_no_trapping/frames/frame0.png")



# # Manual invasion percolation algorithm
# Pinv = np.zeros(pn.Np, dtype=bool)  # Pre-allocate array for storing pore invasion state
# Tinv = np.zeros(pn.Nt, dtype=bool)  # Pre-allocate array for storing throat invasion state
# Invader = 1  # Define the invading pore
# Pinv[Invader] = True
# Ts = pn.find_neighbor_throats(pores=[Invader])

# q = [(air['throat.entry_pressure'][i], i) for i in Ts]
# print(q)
# Tids = []
# q = []
# for i in Ts:
#     q.append((air['throat.entry_pressure'][i], i))
#     Tids.append(i)

# print(q,Tids)
# op.visualization.plot_coordinates(pn, size_by=pn['pore.diameter'], markersize=50, c='b', ax=ax1)
# op.visualization.plot_connections(pn, Ts, size_by=pn['throat.diameter'], linewidth=3,label_by=Tids, c='b', ax=ax1);
# fig1.savefig(path+"/results/Network.png")

# hq.heapify(q)
# print(q)
# T = hq.heappop(q)
# print(T)

# Tinv[T[1]] = True
# P_new = pn.conns[T[1]]
# print(P_new)

# fig2,ax2 = plt.subplots()

# k=0
# buf_dx = io.BytesIO()

# os.makedirs(path+"/results/videos/ip_manual/frames/", exist_ok=True)
# files = glob.glob(path+"/results/videos/ip_manual/frames/*")
# for f in files:
#     os.remove(f)

# image_files = []

# op.visualization.plot_coordinates(pn, size_by=pn['pore.diameter'], markersize=50, c='b', ax=ax2);
# op.visualization.plot_connections(pn, pn.throats() ,size_by=pn['throat.diameter'], linewidth=3, c='b' ,ax=ax2);
# for _ in range(100):
#     T = hq.heappop(q)
#     # If next throat in q is the same index, pop until it's not
#     while q[0][1] == T[1]:
#         # print(f"popping duplicate throat {T[1]}")
#         hq.heappop(q)
#     Tinv[T[1]] = True
#     P_new = pn.find_connected_pores(throats=T[1], flatten=True)
#     P_next = P_new[~Pinv[P_new]]
#     Pinv[P_next] = True
#     T_new = pn.find_neighbor_throats(pores=P_next)
#     T_next = T_new[~Tinv[T_new]]
#     for i in T_next:
#         hq.heappush(q, (air['throat.entry_pressure'][i], i))
#     op.visualization.plot_connections(pn, Tinv ,size_by=pn['throat.diameter'], linewidth=3, c='r' ,ax=ax2);
#     op.visualization.plot_coordinates(pn, Pinv, size_by=pn['pore.diameter'], markersize=50, c='r', ax=ax2);
#     k+=1
#     fig2.savefig(path+"/results/videos/ip_manual/frames/frame"+str(k)+".png")
#     image_files.append(path+"/results/videos/ip_manual/frames/frame"+str(k)+".png")

# clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
# clip.write_videofile(path+"/results/videos/ip_manual/saturation.mp4")


fig3,ax3 = plt.subplots()
ip = op.algorithms.InvasionPercolation(network=pn, phase=air)
np.random.seed(5)
Inlet = pn.pores('left')
Outlet = pn.pores('right')
pn['pore.volume'][Inlet] = 0.0
ip.set_inlet_BC(pores=Inlet)
ip.set_outlet_BC(pores=Outlet, mode='overwrite')
ip.run()
data_ip_no_trapping = ip.pc_curve()

os.makedirs(path+"/results/videos/ip_no_trapping/frames/", exist_ok=True)
files = glob.glob(path+"/results/videos/ip_no_trapping/frames/*")
for f in files:
    os.remove(f)


image_files = []

op.visualization.plot_coordinates(pn, size_by=pn['pore.diameter'], markersize=50, c='b', ax=ax3);
op.visualization.plot_connections(pn, pn.throats() ,size_by=pn['throat.diameter'], linewidth=3, c='b' ,ax=ax3);
op.visualization.plot_coordinates(pn, pn.pores('left'), size_by=pn['pore.diameter'], c='r', ax=ax3)
fig3.savefig(path+"/results/videos/ip_no_trapping/frames/frame0.png")


for sequence in range(1, len(ip['throat.invasion_sequence']), 10):
    inv_throat_pattern = ip['throat.invasion_sequence'] <= sequence
    inv_pore_pattern = ip['pore.invasion_sequence'] <= sequence
    op.visualization.plot_coordinates(pn, inv_pore_pattern, size_by=pn['pore.diameter'], markersize=50, c='r', ax=ax3)
    op.visualization.plot_connections(pn, inv_throat_pattern,size_by=pn['throat.diameter'], linewidth=3, c='r' ,ax=ax3);
    # ax3.legend(['water_pores', 'water_throats', 'gas_pores', 'gas_throats'], loc='best')
    fig3.savefig(path+"/results/videos/ip_no_trapping/frames/frame"+str(sequence)+".png")
    image_files.append(path+"/results/videos/ip_no_trapping/frames/frame"+str(sequence)+".png")

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(path+"/results/videos/ip_no_trapping/saturation_ip_no_trapping.mp4")

ip.apply_trapping()
print(len(ip['throat.invasion_sequence']))
data_ip_trapping = ip.pc_curve()

os.makedirs(path+"/results/videos/ip_trapping/frames/", exist_ok=True)
files = glob.glob(path+"/results/videos/ip_trapping/frames/*")
for f in files:
    os.remove(f)

image_files = []

fig4,ax4 = plt.subplots()

op.visualization.plot_coordinates(pn, size_by=pn['pore.diameter'], markersize=50, c='b', ax=ax4);
op.visualization.plot_connections(pn, pn.throats() ,size_by=pn['throat.diameter'], linewidth=3, c='b' ,ax=ax4);
op.visualization.plot_coordinates(pn, pn.pores('left'), size_by=pn['pore.diameter'], c='r', ax=ax4)
fig4.savefig(path+"/results/videos/ip_trapping/frames/frame0.png")

for sequence in range(1, len(ip['throat.invasion_sequence']), 10):
    inv_throat_pattern = ip['throat.invasion_sequence'] <= sequence
    inv_pore_pattern = ip['pore.invasion_sequence'] <= sequence
    op.visualization.plot_coordinates(pn, inv_pore_pattern, size_by=pn['pore.diameter'], markersize=50, c='r', ax=ax4);
    op.visualization.plot_connections(pn, inv_throat_pattern,size_by=pn['throat.diameter'], linewidth=3, c='r' ,ax=ax4);
    # ax4.legend(['water_pores', 'water_throats', 'gas_pores', 'gas_throats'], loc='best')
    fig4.savefig(path+"/results/videos/ip_trapping/frames/frame"+str(sequence)+".png")
    image_files.append(path+"/results/videos/ip_trapping/frames/frame"+str(sequence)+".png")

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(path+"/results/videos/ip_trapping/saturation_ip_trapping.mp4")


fig5,ax5 = plt.subplots()
line1, = ax5.semilogx(data_ip_no_trapping.pc, data_ip_no_trapping.snwp*100, 'b-', label='without trapping')
line2, = ax5.semilogx(data_ip_trapping.pc, data_ip_trapping.snwp*100, 'r--', label='with trapping', linewidth=1)
ax5.set_xlabel('Capillary Pressure [Pa]')
ax5.set_ylabel('Non-Wetting Phase Saturation[%]')
ax5.legend(handles=[line1, line2], loc='best');
ax5.set_title('Cubic Network with Invasion Percolation')
ax5.set_xlim(1e5, 1e8)
ax5.set_ylim(0, 100)
fig5.savefig(path+"/results/ip_CapillaryPressureXSaturation.png")

