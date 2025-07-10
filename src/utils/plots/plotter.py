from .layout import get_plot_layout
import matplotlib.pyplot as plt

class Plotter2D:
    """Generic 2D plotter with layout support."""
    def __init__(self, layout = "default", **overrides):
        if isinstance(layout, str):
            self.layout = get_plot_layout(layout)
        else:
            self.layout = layout
            
        # Apply overrides
        self.layout.update(overrides)
        
        self.fig, self.ax = plt.subplots(figsize=self.layout.get("figsize", (10, 10)))

    def apply_layout(self):
        """Apply all layout settings to the axes."""
        ax = self.ax
        layout = self.layout
        
        ax.set_title(
            layout.get("title", ""), 
            fontsize=layout.get("title_fontsize", 16)
        )
        ax.set_xlabel(layout.get("xlabel", ""))
        ax.set_ylabel(layout.get("ylabel", ""))
        ax.grid(layout.get("grid", True))
        ax.set_xlim(layout.get("xmin"), layout.get("xmax"))
        ax.set_ylim(layout.get("ymin"), layout.get("ymax"))
        
        if layout.get("sci_notation", False):
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            
    def save(self, path):
        self.fig.savefig(path)
        plt.close(self.fig)
        
class Plotter3D(Plotter2D):
    """3D plotter with layout support."""
    def __init__(self, layout = "pore_network_3d", **overrides):
        super().__init__(layout, **overrides)

        self.ax = self.fig.add_subplot(projection='3d')

    def apply_layout(self):
        """Apply all 2D and 3D layout settings to the axes."""
        super().apply_layout()
        ax = self.ax
        layout = self.layout
        ax.set_zlabel(layout.get("zlabel", ""))
        ax.set_zlim(layout.get("zmin"), layout.get("zmax"))
        ax.view_init(
            elev=layout.get("elev", 15),
            azim=layout.get("azim", -60)
        )