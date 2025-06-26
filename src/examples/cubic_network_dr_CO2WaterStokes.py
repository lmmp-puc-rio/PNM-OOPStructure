# importing Packages
import openpnm as op
import numpy as np
import os as os
import glob
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip
from rich.progress import Progress  
import plotly.graph_objects as go
op.visualization.set_mpl_style()
path = os.path.dirname(__file__)


def create_fig(xrange,yrange):
    fig = go.Figure()
        
    fig.update_layout(autosize=False, 
                    width=1000,
                    height=1000,
                    xaxis=dict(range=xrange), 
                    yaxis = dict( range=yrange,
                                scaleanchor="x",
                                scaleratio=1,
                                constrain='domain'),
                    template="simple_white"
                )
    fig.update_xaxes(
        showline=True,
        mirror=True,
        linecolor="black",
        linewidth=2,
        ticks="outside",
        showexponent = 'all',
        exponentformat = 'e',
        ticklen=6,
    )
    
    fig.update_yaxes(
    showline=True,
    mirror=True,
    linecolor="black",
    linewidth=2,
    ticks="outside",
    showexponent = 'all',
    exponentformat = 'e',
    ticklen=6,
    )
    return fig

def pn_plot(network,pores=None,throats=None,pore_colors=None, throat_colors=None,linewidth = 2,markersize=20,fig = None):
    if pores is None:
        pores = network.pores()
    if throats is None:
        throats = network.throats()
    dim = op.topotools.dimensionality(network)
    
    pores_bool = np.isin(network.pores(), pores)
    throats_bool = np.isin(network.throats(), throats)
    
    pore_diam = network['pore.diameter'][pores_bool]
    markersize = pore_diam / pore_diam.max() * markersize
    
    thr_diam = network['throat.diameter'][throats_bool]
    linewidth = thr_diam / thr_diam.max() * linewidth
    
    if fig is None:
        #layout
        X, Y, Z = network['pore.coords'].T
        margin = max(np.max(X), np.max(Y)) * 0.05
        xmin = min(X) - margin
        xmax = max(X) + margin
        ymin = min(Y) - margin
        ymax = max(Y) + margin
        fig = create_fig(xrange=[xmin,xmax],yrange=[ymin,ymax])
    else:
        fig.data = () 
        
    
    #Throats
    Ps = np.unique(network['throat.conns'][throats])
    X, Y, Z = network['pore.coords'][Ps].T
    xyz = network["pore.coords"][:, dim]
    P1, P2 = network["throat.conns"][throats].T
    throat_pos = np.column_stack((xyz[P1], xyz[P2])).reshape((throats.size, 2, dim.sum()))
    
    x_t = throat_pos[:, :, 0].reshape(-1, 2)
    y_t = throat_pos[:, :, 1].reshape(-1, 2)

    for (x12, y12, w, c) in zip(x_t,y_t,linewidth,throat_colors):
        fig.add_trace(go.Scatter(
            x=x12,
            y=y12,
            mode='lines',
            line=dict(width=w,color=c),
            showlegend=False
        ))
        
    #Pores
    X, Y, Z = network['pore.coords'][pores].T

    fig.add_trace(go.Scatter(
        x=X, 
        y=Y, 
        mode='markers',
        showlegend=False,
        marker=dict(size=list(markersize),color=list(pore_colors), opacity=1), 
        opacity=1))
    
    return fig

npores = 10
np.random.seed(15)
fps = 10
trapping = 'trapping'  # Options: 'trapping', 'no_trapping'
Lc = 1e-6
spacing = 1e-4
# Create a cubic network
pn = op.network.Cubic(shape=[npores, npores, 1], spacing=spacing)

msize = 100  # marker size for visualization
lwidth = 3  # line width for visualization
azim = -60
elev = 15

max_lenght = npores*spacing

pn_dim = '2D'
if op.topotools.dimensionality(pn).sum() == 3:
    pn_dim = '3D'

pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
pn.regenerate_models()
pn['pore.diameter'] = np.random.rand(pn.Np)*Lc
pn['throat.diameter'] = np.random.rand(pn.Nt)*Lc
co2 = op.phase.Air(network=pn,name='CO2')
co2['pore.surface_tension'] = 0.023 # Surface tension of CO2 (N/m)
co2['pore.contact_angle'] = 150.0

co2.add_model_collection(op.models.collections.phase.air)
co2.add_model_collection(op.models.collections.physics.basic)
co2.regenerate_models()
water = op.phase.Water(network=pn,name='water')
water['throat.diffusivity'] = 1e-8  # Water's diffusivity in scCO2 ( m²/s )
water['pore.viscosity'] = 1e-5  # Water's viscosity ( Pas )
water['pore.surface_tension'] = 0.023 # Surface tension of CO2 (N/m)
water['pore.contact_angle'] = 30.0
water.add_model_collection(op.models.collections.phase.water)
water.add_model_collection(op.models.collections.physics.basic)
water.regenerate_models()

graph_path = os.path.join(path, 'results', f'DRRelPerm{pn_dim}_{trapping}', 'graphs')
video_path = os.path.join(path, 'results', f'DRRelPerm{pn_dim}_{trapping}', 'videos')
frame_path = os.path.join(video_path, f'frames_{npores}_pores')

os.makedirs(graph_path, exist_ok=True)
os.makedirs(frame_path , exist_ok=True)

fig0,ax0 = plt.subplots()
ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
op.visualization.plot_coordinates(pn, size_by=pn['pore.diameter'], markersize=msize,  c='b',alpha=0.8, ax=ax0)
op.visualization.plot_connections(pn, size_by=pn['throat.diameter'], linewidth=lwidth, c='b',alpha=0.8, ax=ax0)
fig0.savefig(os.path.join(graph_path, f'Network{pn_dim}_CO2WaterStokes_{trapping}{npores}.png'))
dr = op.algorithms.Drainage(network=pn, phase=co2)
Inlet = pn.pores('left')
Outlet = pn.pores('right')
dr.set_inlet_BC(pores=Inlet)
pn['pore.volume'][Inlet] = 0.0
dr.run(pressures=200)

Snwp_num=30
flow_in = pn.pores('left')
flow_out = pn.pores('right')

def sat_occ_update(network, nwp, wp, dr, i):
    r"""
        Calculates the saturation of each phase using the invasion
        sequence from either invasion percolation.
        Parameters
        ----------
        network: network
        nwp : phase
            non-wetting phase
        wp : phase
            wetting phase
        dr : DR
            drainage (ran before calling this function)
        i: int
            The invasion_sequence limit for masking pores/throats that
            have already been invaded within this limit range. The
            saturation is found by adding the volume of pores and thorats
            that meet this sequence limit divided by the bulk volume.
    """
    pore_mask = dr['pore.invasion_sequence'] < i
    throat_mask = dr['throat.invasion_sequence'] < i
    sat_p = np.sum(network['pore.volume'][pore_mask])
    sat_t = np.sum(network['throat.volume'][throat_mask])
    sat1 = sat_p + sat_t
    bulk = network['pore.volume'].sum() + network['throat.volume'].sum()
    sat = sat1/bulk
    nwp['pore.occupancy'] = pore_mask
    nwp['throat.occupancy'] = throat_mask
    wp['throat.occupancy'] = 1-throat_mask
    wp['pore.occupancy'] = 1-pore_mask
    return sat

def Rate_calc(network, phase, left, outlet, conductance):
    phase.regenerate_models()
    St_p = op.algorithms.StokesFlow(network=network, phase=phase)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=left, values=1)
    St_p.set_value_BC(pores=outlet, values=0)
    St_p.run()
    val = np.abs(St_p.rate(pores=left, mode='group'))
    return val

# Define Multiphase Conductance model
model_mp_cond = op.models.physics.multiphase.conduit_conductance
co2.add_model(model=model_mp_cond, propname='throat.conduit_hydraulic_conductance',
              throat_conductance='throat.hydraulic_conductance', mode='medium', regen_mode='deferred')
water.add_model(model=model_mp_cond, propname='throat.conduit_hydraulic_conductance',
              throat_conductance='throat.hydraulic_conductance', mode='medium', regen_mode='deferred')


# Max Invasion Sequence
max_seq = np.max(dr['throat.invasion_sequence'])

start = 0 # max_seq//Snwp_num
stop = max_seq+1
step = max_seq//Snwp_num
print("Start: ", start, " Stop: ", stop, " Step: ", step)
Snwparr = []
relperm_nwp = []
relperm_wp = []

## Apply Trapping
if trapping == 'trapping':
    dr.set_outlet_BC(pores=pn.pores('right'), mode='overwrite')
    dr.apply_trapping()

tmask = np.isfinite(dr['throat.invasion_sequence'])

max_seq = np.max(dr['throat.invasion_sequence'][tmask])
start = 0 # max_seq//Snwp_num
stop = max_seq+1
step = max_seq//Snwp_num
print("Start: ", start, " Stop: ", stop, " Step: ", step)
Snwparr = []
relperm_nwp = []
relperm_wp = []

# Loop through invasion sequence to calculate saturation and relative permeability
for i in range(start, int(stop), int(step)):
    co2.regenerate_models()
    water.regenerate_models()
    sat = sat_occ_update(network=pn, nwp=co2, wp=water, dr=dr, i=i)
    Snwparr.append(sat)
    Rate_abs_nwp = Rate_calc(pn, co2, flow_in, flow_out, conductance = 'throat.hydraulic_conductance')
    Rate_abs_wp = Rate_calc(pn, water, flow_in, flow_out, conductance = 'throat.hydraulic_conductance')
    Rate_enwp = Rate_calc(pn, co2, flow_in, flow_out, conductance = 'throat.conduit_hydraulic_conductance')
    Rate_ewp = Rate_calc(pn, water, flow_in, flow_out, conductance = 'throat.conduit_hydraulic_conductance')
    relperm_nwp.append(Rate_enwp/Rate_abs_nwp)
    relperm_wp.append(Rate_ewp/Rate_abs_wp)


left = pn.pores('left')
Domain = pn.pores('left', mode='not')
k=0

data_dr_trapping = dr.pc_curve()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(Snwparr, relperm_nwp, '*-', label='Kr_nwp')
ax1.plot(Snwparr, relperm_wp, 'o-', label='Kr_wp')
ax1.set_xlabel('Snwp')
ax1.set_ylabel('Kr')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.05)
ax1.set_title(f'Relative Permeability in x direction - {trapping.replace("_"," ")}', fontsize=16)
ax1.legend()
fig1.savefig(os.path.join(graph_path, f'DRRelPerm{pn_dim}_{trapping}.png'))

# Making fames of Invasion Percolation saturation field
files = glob.glob(os.path.join(frame_path, '*'))
for f in files:
    os.remove(f)

image_files = []

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection = '3d') if pn_dim == '3D' else fig2.add_subplot(111)
op.visualization.plot_coordinates(pn, pn.pores('left',mode= 'nor'),size_by=pn['pore.diameter'], markersize=msize,alpha=0.3, c='b', ax=ax2)
op.visualization.plot_connections(pn, pn.throats() ,size_by=pn['throat.diameter'], linewidth=lwidth, c='b' ,alpha=0.3,ax=ax2)
op.visualization.plot_coordinates(pn, pn.pores('left'), size_by=pn['pore.diameter'], markersize=msize, c='r',alpha=0.5,ax=ax2)
fig2.set_size_inches(npores, npores)
ax2.set_aspect('auto')
ax2.set_title(f'Pressure = {(data_dr_trapping.pc[k])} Pa',fontsize=16)
ax2.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
ax2.grid(False)
if pn_dim == '3D':
    ax2.set_zlim((0, max_lenght))
    ax2.view_init(elev=elev, azim=azim)
fig2.savefig(os.path.join(frame_path,'frame0.png'))
invasion_sequence = np.unique(dr['throat.invasion_sequence'][np.isfinite(dr['throat.invasion_sequence'])])

steps = 20
single_step = len(invasion_sequence)//20

with Progress() as p:
    t = p.add_task("Generating Video:", total=steps)
    for sequence in invasion_sequence[::single_step]:
        invasion_pressure = max(dr['throat.invasion_pressure'][dr['throat.invasion_sequence'] == sequence])/1000
        k += 1
        inv_throat_pattern = dr['throat.invasion_sequence'] <= sequence
        inv_pore_pattern = dr['pore.invasion_sequence'] <= sequence
        op.visualization.plot_connections(pn, inv_throat_pattern,size_by=pn['throat.diameter'],alpha=0.8, linewidth=lwidth, c='r' ,ax=ax2)
        op.visualization.plot_coordinates(pn, inv_pore_pattern, size_by=pn['pore.diameter'], markersize=msize, c='r',ax=ax2)
        ax2.set_aspect('auto')
        ax2.set_title(f'Pressure = {invasion_pressure:.2f} kPa',fontsize=16)
        ax2.set_xlim((0, max_lenght))
        ax2.set_ylim((0, max_lenght))
        if pn_dim == '3D':
            ax2.set_zlim((0, max_lenght))
            ax2.view_init(elev=elev, azim=azim)
        fig2.savefig(os.path.join(frame_path,f'frame{k}.png'))
        image_files.append(os.path.join(frame_path,f'frame{k}.png'))
        p.update(t, advance=1)
        non_invaded_pores = dr['pore.invasion_sequence'] > sequence
        non_invaded_throats = dr['throat.invasion_sequence'] > sequence
    t = p.add_task("Irreducible Water:", total=10)
    if trapping == 'trapping':
        for j in range(0, 10, 1):
            for collection in ax2.collections:
                collection.remove()
            op.visualization.plot_coordinates(pn, non_invaded_pores, size_by=pn['pore.diameter'], alpha=j/10,markersize=msize, c='b',ax=ax2)
            op.visualization.plot_connections(pn, non_invaded_throats ,size_by=pn['throat.diameter'], linewidth=lwidth,alpha=j/10, c='b' ,ax=ax2)
            op.visualization.plot_connections(pn, inv_throat_pattern,size_by=pn['throat.diameter'], linewidth=lwidth,alpha=1-j/10, c='r' ,ax=ax2)
            op.visualization.plot_coordinates(pn, inv_pore_pattern, size_by=pn['pore.diameter'], markersize=msize,alpha=1-j/10, c='r',ax=ax2)
            ax2.set_title(f'Pressure = {invasion_pressure} kPa',fontsize=16)
            ax2.set_aspect('auto')
            ax2.set_xlim((0, max_lenght))
            ax2.set_ylim((0, max_lenght))
            if pn_dim == '3D':
                ax2.set_zlim((0, max_lenght))
                ax2.view_init(elev=elev, azim=azim)
            k += 1
            fig2.savefig(os.path.join(frame_path,f'frame{k}.png'))
            image_files.append(os.path.join(frame_path,f'frame{k}.png'))
            p.update(t, advance=1)
    if pn_dim == '3D':
        t = p.add_task("Rotação:", total=36)
        for l in range(0, 36, 1):
            ax2.view_init(elev=elev, azim=azim+l*10)
            k += 1
            fig2.savefig(os.path.join(frame_path,f'frame{k}.png'))
            image_files.append(os.path.join(frame_path,f'frame{k}.png'))
            p.update(t, advance=1)

#IMBIBITION

water['throat.entry_pressure'] = water['throat.entry_pressure'] *-1
im = op.algorithms.Drainage(network=pn, phase=water)
im.set_inlet_BC(pores=Inlet,mode= 'overwrite')
im.set_outlet_BC(pores=Outlet, mode='overwrite')
im['pore.invaded'] = dr['pore.trapped'].copy()
im['throat.invaded'] = dr['throat.trapped'].copy()

hi = water['throat.entry_pressure'].max()
low = 0.80*water['throat.entry_pressure'].min()
steps = 300
x = np.linspace(0,1,steps)
imb_pressures = low * (hi/low) ** x  

water_ic_pore = im['pore.invaded'].copy()
water_ic_throat = im['throat.invaded'].copy()

im.run(pressures=imb_pressures)


fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection = '3d') if pn_dim == '3D' else fig3.add_subplot(111)

fig3.set_size_inches(npores, npores)
ax3.set_aspect('auto')
ax3.grid(False)
if pn_dim == '3D':
    ax3.set_zlim((0, max_lenght))
    ax3.view_init(elev=elev, azim=azim)

k += 1

throats_water = pn.throats()[water_ic_throat]
pores_water = pn.pores()[water_ic_pore]

throats_air = pn.throats()[~water_ic_throat]
pores_air = pn.pores()[~water_ic_pore]

op.visualization.plot_connections(pn, throats_water,size_by=pn['throat.diameter'],alpha=0.3, linewidth=lwidth, c='b' ,ax=ax3)
op.visualization.plot_coordinates(pn, pores_water , size_by=pn['pore.diameter'],alpha=0.3, markersize=msize, c='b',ax=ax3)
op.visualization.plot_connections(pn, throats_air,size_by=pn['throat.diameter'],alpha=0.3, linewidth=lwidth, c='r' ,ax=ax3)
op.visualization.plot_coordinates(pn, pores_air, size_by=pn['pore.diameter'],alpha=0.3, markersize=msize, c='r',ax=ax3)

fig3.savefig(os.path.join(frame_path,f'frame{k}.png'))
image_files.append(os.path.join(frame_path,f'frame{k}.png'))

water_color = "rgba(0,0,255,1)"
air_color = "rgba(255,0,0,1)"

for sequence in np.unique(im['throat.invasion_sequence'][np.isfinite(im['throat.invasion_sequence'])]):
    k += 1
    # for collection in ax3.collections:
    #         collection.remove()
    inv_throat_pattern = im['throat.invasion_sequence'] <= sequence
    inv_pore_pattern = im['pore.invasion_sequence'] <= sequence
    
    new_pores = np.setdiff1d(pn.pores()[inv_pore_pattern], pores_water)
    new_throats = np.setdiff1d(pn.throats()[inv_throat_pattern], throats_water)
    
    throats_water = np.union1d(throats_water,new_throats)
    pores_water = np.union1d(pores_water,new_pores)
    
    # throats_air = np.setdiff1d(throats_air, new_throats)
    # pores_air = np.setdiff1d(pores_air, new_pores)
    
    pores_bool = np.isin(pn.pores(), pores_water)
    throats_bool = np.isin(pn.throats(), throats_water)
    
    pore_colors = np.where(pores_bool, water_color, air_color)
    throat_colors = np.where(throats_bool, water_color, air_color)
    
    fig = pn_plot(pn,pore_colors=pore_colors,throat_colors=throat_colors)
    
    fig.write_image(os.path.join(frame_path,f'frame{k}.png'),width=1000, height=1000,scale=1)
    
    # op.visualization.plot_coordinates(pn, pores_water, size_by=pn['pore.diameter'],alpha=0.6, markersize=msize, c='b',ax=ax3)
    # op.visualization.plot_coordinates(pn, pores_air, size_by=pn['pore.diameter'],alpha=0.6, markersize=msize, c='r',ax=ax3)
    
    # op.visualization.plot_connections(pn, throats_water ,size_by=pn['throat.diameter'],alpha=0.6, linewidth=lwidth, c='b' ,ax=ax3)
    # op.visualization.plot_connections(pn, throats_air ,size_by=pn['throat.diameter'],alpha=0.6, linewidth=lwidth, c='r' ,ax=ax3)
    
    # fig3.savefig(os.path.join(frame_path,f'frame{k}.png'))
    image_files.append(os.path.join(frame_path,f'frame{k}.png'))

# clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
# clip.write_videofile(os.path.join(video_path,f'saturation_DRRelPerm{pn_dim}_{trapping}_{npores}.mp4'))