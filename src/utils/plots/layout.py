PORE_NETWORK_2D_LAYOUT = {
        "figsize": (10, 10),
        "title": "Pore Network",
        "title_fontsize": 16,
        "grid": False,
        "sci_notation": True,
        "xlabel": "X [m]",
        "ylabel": "Y [m]",
    }

PORE_NETWORK_3D_LAYOUT = {
        "figsize": (10, 10),
        "title": "Pore Network",
        "title_fontsize": 16,
        "grid": False,
        "sci_notation": True,
        "elev": 15,
        "azim": -60,
        "xlabel": "X [m]",
        "ylabel": "Y [m]",
        "zlabel": "Z [m]",
    }

INVASION_2D_LAYOUT = {
        "figsize": (10, 10),
        "title": "Invasion Percolation",
        "title_fontsize": 16,
        "grid": False,
        "sci_notation": True,
        "xlabel": "X [m]",
        "ylabel": "Y [m]",
    }

INVASION_3D_LAYOUT = {
        "figsize": (10, 10),
        "title": "Invasion Percolation",
        "title_fontsize": 16,
        "grid": False,
        "sci_notation": True,
        "elev": 15,
        "azim": -60,
        "xlabel": "X [m]",
        "ylabel": "Y [m]",
        "zlabel": "Z [m]",
    }

RELATIVE_PERMEABILITY_LAYOUT = {
        "figsize": (6, 6),
        "title": "Relative Permeability",
        "title_fontsize": 16,
        "xlabel": "Snwp [%]",
        "ylabel": "Kr",
        "grid": True,
        "sci_notation": False,
        "xmin": 0,
        "xmax": 100,
        "ymin": -0.01,
        "ymax": 1.05,
        "legend": True,
    }

def get_plot_layout(plot_name: str):
    """Retrieve a predefined layout configuration by name.
    
    Args:
        plot_name: Key mapping to a layout constant (e.g., 'pore_network_3d')
    
    Returns:
        Dictionary of layout parameters
        
    Raises:
        KeyError: If plot_name is invalid
    """
    layout_map = {
        'pore_network_2d': PORE_NETWORK_2D_LAYOUT,
        'pore_network_3d': PORE_NETWORK_3D_LAYOUT,
        'invasion_2d': INVASION_2D_LAYOUT,
        'invasion_3d': INVASION_3D_LAYOUT,
        'relative_permeability': RELATIVE_PERMEABILITY_LAYOUT,
    }
    
    try:
        return layout_map[plot_name]
    except KeyError:
        available = list(layout_map.keys())
        raise KeyError(
            f"Invalid plot_name '{plot_name}'. Available layouts: {available}"
        )

