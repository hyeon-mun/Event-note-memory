import numpy as np
NAMES = {"red": (220, 10, 10), "green": (10, 225, 20), "blue": (5, 5, 230), "yellow": (238, 220, 14),
         "magenta": (222, 8, 228), "cyan": (10, 220, 220), "orange": (240, 140, 10), "white": (230, 230, 230),
         "purple": (120, 20, 200), "maroon": (160, 10, 1), "teal": (8, 167, 166)}


def color_names(C):
    return [min(NAMES, key=lambda n: np.sum((np.array(NAMES[n]) - c) ** 2)) for c in C]
