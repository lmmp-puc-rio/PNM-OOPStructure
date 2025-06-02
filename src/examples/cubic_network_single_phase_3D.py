# importing Packages
import openpnm as op
import numpy as np
import os as os
import glob
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip

op.visualization.set_mpl_style()
path = os.path.dirname(__file__)

# Setup Necessary Objects
fps = 5

np.random.seed(10)
np.set_printoptions(precision=5)

n_pores = 15
spacing = 1e-6

msize = 50  # marker size for visualization
lwidth = 3  # line width for visualization
azim = -60
elev = 15

# Create a cubic network
pn = op.network.Demo(shape=[n_pores, n_pores, n_pores], spacing=spacing)
max_lenght = n_pores*spacing


os.makedirs(path+"/results/videos/singlePhase/frames/", exist_ok=True)
files = glob.glob(path+"/results/videos/singlePhase/frames/*")
for f in files:
    os.remove(f)

phase = op.phase.Phase(network=pn)
phase['pore.viscosity']=1.0 # Viscosity in Pa.s
phase.add_model_collection(op.models.collections.physics.basic)
phase.regenerate_models()

fig0,ax0 = plt.subplots()
op.visualization.plot_coordinates(pn, size_by=pn['pore.diameter'], markersize=msize,  c='b',alpha=0.8, ax=ax0)
op.visualization.plot_connections(pn, size_by=pn['throat.diameter'], linewidth=lwidth, c='b',alpha=0.8, ax=ax0)
fig0.savefig(path+"/results/Network3D_singlePhase.png")

inlet = pn.pores('left')
outlet = pn.pores('right')
flow = op.algorithms.StokesFlow(network=pn, phase=phase)
flow.set_value_BC(pores=inlet, values=1)
flow.set_value_BC(pores=outlet, values=0)
flow.run()
phase.update(flow.soln)

# NBVAL_IGNORE_OUTPUT
Q = flow.rate(pores=inlet, mode='group')[0]
A = op.topotools.get_domain_area(pn, inlets=inlet, outlets=outlet)
L = op.topotools.get_domain_length(pn, inlets=inlet, outlets=outlet)
# K = Q * L * mu / (A * Delta_P) # mu and Delta_P were assumed to be 1.
K = Q * L / A
kmD = K/0.98e-12*1000

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
op.visualization.plot_coordinates(pn, size_by=pn['pore.diameter'], markersize=msize, color_by=phase['pore.pressure'],alpha=0.3, ax=ax1)
op.visualization.plot_connections(pn, size_by=pn['throat.diameter'], linewidth=lwidth, color_by=phase['throat.pressure'],alpha=0.3, ax=ax1)
ax1.set_aspect('auto')
ax1.set_xlim(0, n_pores*spacing)
ax1.set_ylim(0, n_pores*spacing)
ax1.set_zlim(0, n_pores*spacing)
ax1.set_xlabel('X [m]')
ax1.set_ylabel('Y [m]')
ax1.set_zlabel('Z [m]')
fig1.set_size_inches(n_pores,n_pores)
ax1.view_init(elev=elev, azim=azim)
ax1.set_title('Pressure Field:' + f' K = {K/0.98e-12*1000:.5f} mD',fontsize=2*n_pores)
fig1.savefig(path+"/results/singlePhase_PressureField.png")
