# importing Packages
import openpnm as op
import numpy as np
import os as os
import glob
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip

op.visualization.set_mpl_style()
path = os.path.dirname(__file__)


np.random.seed(5)








n_pores = 25
np.random.seed(5)
spacing = 1e-4
fps = 10
Lc = 1e-6
pn = op.network.Cubic(shape=[n_pores, n_pores, 1], spacing=spacing)
pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
pn.regenerate_models()
max_lenght = n_pores*spacing

pn['pore.diameter'] = np.random.rand(pn.Np)*Lc
pn['throat.diameter'] = np.random.rand(pn.Nt)*Lc
water = op.phase.Water(network=pn,name='water')
air = op.phase.Air(network=pn)
water.add_model_collection(op.models.collections.phase.water)
water.add_model_collection(op.models.collections.physics.basic)
water.regenerate_models()


os.makedirs(path+"/results/videos/drainage_manual/frames/", exist_ok=True)
files = glob.glob(path+"/results/videos/drainage_manual/frames/*")
for f in files:
    os.remove(f)


fig0,ax0 = plt.subplots()
op.visualization.plot_coordinates(pn, size_by=pn['pore.diameter'], markersize=50, c='b', ax=ax0)
op.visualization.plot_connections(pn, size_by=pn['throat.diameter'], linewidth=3, c='b', ax=ax0)
fig0.savefig(path+"/results/Network.png")

## Define capillary pressure as 70% of throats mean entry pressure
washAir = op.models.physics.capillary_pressure.washburn
air['pore.contact_angle'] = 120
air['pore.surface_tension'] = 0.072
air.add_model(propname='throat.entry_pressure',
              model=washAir, 
              surface_tension='throat.surface_tension',
              contact_angle='throat.contact_angle',
              diameter='throat.diameter',)
Pc = air['throat.entry_pressure'].mean()*0.7 
Ts = 1.0*(air['throat.entry_pressure'] < Pc)



# # Manual Implementation of Drainage Process
# from scipy.sparse import csgraph as csg
# am = pn.create_adjacency_matrix(weights=Ts, fmt='csr', triu=False, drop_zeros=True)
# clusters = csg.connected_components(am, directed=False)[1]
# print(clusters)
# fig1,ax1 = plt.subplots()
# op.visualization.plot_coordinates(network=pn, color_by=clusters, s=40, cmap=plt.cm.nipy_spectral,ax=ax1)
# op.visualization.plot_connections(network=pn, c='lightgrey', ax=ax1)
# fig1.savefig(path+"/results/drn_manual_clusters.png")

# invaded_pores = np.isin(clusters, clusters[pn.pores('left')])
# fig2,ax2 = plt.subplots()
# op.visualization.plot_coordinates(network=pn, color_by=invaded_pores, s=40, cmap=plt.cm.viridis, ax=ax2)
# op.visualization.plot_connections(network=pn, c='lightgrey', ax=ax2)
# fig2.savefig(path+"/results/drn_manual_invaded_pores.png")

# Using Drainage Algorithm
drn = op.algorithms.Drainage(network=pn, phase=air)
np.random.seed(5)
Inlet = pn.pores('left')
Outlet = pn.pores('right')
pn['pore.volume'][Inlet] = 0.0
drn.set_inlet_BC(pores=Inlet)
# drn.set_outlet_BC(pores=Outlet, mode='overwrite')
drn.run(pressures=100)

data_drn_no_trapping = drn.pc_curve(pressures=100)

os.makedirs(path+"/results/videos/drn_no_trapping/frames/", exist_ok=True)
files = glob.glob(path+"/results/videos/drn_no_trapping/frames/*")
for f in files:
    os.remove(f)
image_files = []

fig3,ax3 = plt.subplots()
op.visualization.plot_coordinates(pn, pn.pores('left'), size_by=pn['pore.diameter'], markersize=50, c='r',ax=ax3)
op.visualization.plot_coordinates(pn, pn.pores('left',mode= 'nor'),size_by=pn['pore.diameter'], markersize=50, c='b', ax=ax3)
op.visualization.plot_connections(pn, pn.throats() ,size_by=pn['throat.diameter'], linewidth=5, alpha=0.5 , c='b' ,ax=ax3)
ax3.set_xlim((0, max_lenght))
ax3.set_ylim((0, max_lenght))
fig3.set_size_inches(10, 10)
fig3.savefig(path+"/results/videos/drn_no_trapping/frames/frame0.png")
print(len(drn['throat.invasion_sequence']))


for sequence in np.unique(drn['throat.invasion_sequence'][drn['throat.invasion_sequence']!= np.inf]):
    inv_throat_pattern = drn['throat.invasion_sequence'] <= sequence
    inv_pore_pattern = drn['pore.invasion_sequence'] <= sequence
    op.visualization.plot_connections(pn, inv_throat_pattern,size_by=pn['throat.diameter'], linewidth=3, c='r' , alpha=0.5 ,ax=ax3);
    op.visualization.plot_coordinates(pn, inv_pore_pattern, size_by=pn['pore.diameter'], markersize=50, c='r',alpha=0.5 ,ax=ax3);
    ax3.set_aspect('auto')
    ax3.set_xlim((0, max_lenght))
    ax3.set_ylim((0, max_lenght))
    fig3.set_size_inches(10, 10)
    fig3.savefig(path+"/results/videos/drn_no_trapping/frames/frame"+str(sequence)+".png")
    image_files.append(path+"/results/videos/drn_no_trapping/frames/frame"+str(sequence)+".png")

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(path+"/results/videos/drn_no_trapping/saturation_drn_no_trapping.mp4")

# Apply trapping
drn.set_outlet_BC(pores=pn.pores('right'), mode='overwrite')
drn.apply_trapping()
data_drn_trapping = drn.pc_curve(pressures=100)

os.makedirs(path+"/results/videos/drn_trapping/frames/", exist_ok=True)
files = glob.glob(path+"/results/videos/drn_trapping/frames/*")
for f in files:
    os.remove(f)
image_files = []
k=0

fig4,ax4 = plt.subplots()
op.visualization.plot_coordinates(pn, pn.pores('left'), size_by=pn['pore.diameter'], markersize=50, c='r',ax=ax4)
op.visualization.plot_coordinates(pn, pn.pores('left',mode= 'nor'),size_by=pn['pore.diameter'], markersize=50, c='b', ax=ax4)
op.visualization.plot_connections(pn, pn.throats() ,size_by=pn['throat.diameter'], linewidth=5, c='b', alpha=0.5 ,ax=ax4)
fig4.set_size_inches(10, 10)
ax4.set_xlim((0, max_lenght))
ax4.set_ylim((0, max_lenght))
fig4.savefig(path+"/results/videos/drn_trapping/frames/frame0.png")
print(len(drn['throat.invasion_sequence']))
image_files = []

for sequence in np.unique(drn['throat.invasion_sequence'][drn['throat.invasion_sequence']!= np.inf]):
    k += 1
    inv_throat_pattern = drn['throat.invasion_sequence'] <= sequence
    inv_pore_pattern = drn['pore.invasion_sequence'] <= sequence
    op.visualization.plot_connections(pn, inv_throat_pattern,size_by=pn['throat.diameter'], linewidth=3, c='r' , alpha=0.5 ,ax=ax4);
    op.visualization.plot_coordinates(pn, inv_pore_pattern, size_by=pn['pore.diameter'], markersize=50, c='r',alpha=0.5 , ax=ax4);
    ax4.set_aspect('auto')
    ax4.set_xlim((0, max_lenght))
    ax4.set_ylim((0, max_lenght))
    # ax2.set_title('Pressure = '+str(data_drn_trapping.pc[k])+' Pa')
    fig3.set_size_inches(10, 10)
    fig4.savefig(path+"/results/videos/drn_trapping/frames/frame"+str(k)+".png")
    image_files.append(path+"/results/videos/drn_trapping/frames/frame"+str(k)+".png")

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(path+"/results/videos/drn_trapping/saturation_drn_trapping.mp4")


fig5,ax5 = plt.subplots()
fig5.set_size_inches(10,10)
line1, = ax5.semilogx(data_drn_no_trapping.pc, data_drn_no_trapping.snwp*100, 'b-o', label='without trapping', linewidth=1)
line2, = ax5.semilogx(data_drn_trapping.pc, data_drn_trapping.snwp*100, 'r--', label='with trapping',linewidth=1)
ax5.set_xlabel('Capillary Pressure [Pa]')
ax5.set_ylabel('Non-Wetting Phase Saturation[%]')
ax5.legend(handles=[line1, line2], loc='best')
ax5.set_title('Cubic Network with Drainage')
ax5.set_ylim(0, 105)
fig5.savefig(path+"/results/drn_CapillaryPressureXSaturation.png")


# inv_pattern = drn['throat.invasion_pressure'] < 9000
# fig1 = plt.figure(figsize=[6,5]) 
# ax1 = fig1.add_subplot()
# ax1 = op.visualization.plot_coordinates(network=pn, pores=pn.pores('left'), c='r', s=50, ax=ax1)
# ax1 = op.visualization.plot_coordinates(network=pn, pores=pn.pores('left', mode='not'), c='grey', ax=ax1)
# op.visualization.plot_connections(network=pn, throats=inv_pattern, ax=ax1);
# fig1.savefig(path+"/results/drainage/Network"+str(network_size)+".png")


# drn.set_outlet_BC(pores=pn.pores('right'), mode='overwrite')
# drn.apply_trapping()

# inv_pattern = drn['throat.invasion_pressure'] < 9000
# fig2 = plt.figure(figsize=[6,5])
# ax2 = fig2.add_subplot()
# ax = op.visualization.plot_coordinates(network=pn, pores=pn.pores('left'), c='r', s=50, ax=ax2)
# ax = op.visualization.plot_coordinates(network=pn, pores=pn.pores('left', mode='not'), c='grey', ax=ax2)
# op.visualization.plot_connections(network=pn, throats=inv_pattern, ax=ax2);
# fig2.savefig(path+"/results/drainage/Network_trapping"+str(network_size)+".png")

# hg = op.phase.Mercury(network=pn)
# f = op.models.physics.capillary_pressure.washburn
# hg.add_model(propname='throat.entry_pressure',
#              model=f, 
#              surface_tension='throat.surface_tension',
#              contact_angle='throat.contact_angle',
#              diameter='throat.diameter',)
# mip = op.algorithms.Drainage(network=pn, phase=hg)
# mip.set_inlet_BC(pores=pn.pores('surface'))  # mercury invades from all sides
# mip.run()

# # Capilarry Pressure Curve
# data = mip.pc_curve()
# plt.plot(data.pc, data.snwp, 'b-o')
# plt.xlabel('Capillary Pressure [Pa]')
# plt.ylabel('Non-Wetting Phase Saturation');