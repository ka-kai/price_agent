import matplotlib.pyplot as plt

# General plot settings
params = {
    "font.family": "sans-serif",
    "mathtext.fontset": "stixsans",
    "legend.edgecolor": "black",
    "legend.borderaxespad": 0,
    "legend.fancybox": False,  # rectangle instead of rounded corners
    "axes.axisbelow": True,  # grid below the plot
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01
}
plt.rcParams.update(params)

# Colors to be used in plots
dict_colors = {
    "blue":     "#215CAF",  # full
    "red":      "#B7352D",
    "bronze":   "#8E6713",
    "green":    "#627313",
    "petrol":   "#007894",
    "purpur":   "#A7117A",
    "grey":     "#6F6F6F",
    "blue80":   "#4D7DBF",  # 80 %
    "red80":    "#C55D57",
    "bronze80": "#A58542",
    "green80":  "#818F42",
    "petrol80": "#3395AB",
    "purpur80": "#B73B92",
    "grey80":   "#8C8C8C",
}

# Other
cm = 1 / 2.54  # inches
