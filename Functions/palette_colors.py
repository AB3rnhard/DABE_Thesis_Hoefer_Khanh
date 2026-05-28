import numpy as np
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


def palette_colors(palette_hex: list, seq_cmap_mpl, n: int) -> list:
    """Return n hex colours evenly spaced along the active palette."""
    if n <= 1:
        return [palette_hex[len(palette_hex) // 2]]
    if n <= len(palette_hex):
        idx = np.linspace(0, len(palette_hex) - 1, n).astype(int)
        return [palette_hex[i] for i in idx]
    return [mcolors.rgb2hex(seq_cmap_mpl(i / (n - 1))) for i in range(n)]
