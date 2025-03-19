import matplotlib.pyplot as plt
import numpy as np

# Set global style to match Nature's guidelines
plt.rcParams.update({
    'font.family': 'sans-serif',  # Use sans-serif font
    'font.sans-serif': ['Arial', 'Helvetica'],  # Specify Arial or Helvetica
    'font.size': 8,  # Base font size
    'axes.labelsize': 8,  # Axis label font size
    'axes.titlesize': 10,  # Title font size
    'xtick.labelsize': 7,  # X-axis tick label size
    'ytick.labelsize': 7,  # Y-axis tick label size
    'legend.fontsize': 8,  # Legend font size
    'figure.titlesize': 10,  # Figure title size
    'lines.linewidth': 0.75,  # Line width for data
    'axes.linewidth': 0.5,  # Axis line width
    'xtick.major.width': 0.5,  # X-axis major tick width
    'ytick.major.width': 0.5,  # Y-axis major tick width
    'xtick.minor.width': 0.25,  # X-axis minor tick width
    'ytick.minor.width': 0.25,  # Y-axis minor tick width
    'xtick.major.size': 4,  # X-axis major tick length
    'ytick.major.size': 4,  # Y-axis major tick length
    'xtick.minor.size': 2,  # X-axis minor tick length
    'ytick.minor.size': 2,  # Y-axis minor tick length
    'figure.dpi': 300,  # Figure resolution
    'savefig.dpi': 300,  # Save resolution
    'savefig.format': 'tiff',  # Save format (TIFF recommended)
    'savefig.bbox': 'tight',  # Tight bounding box
    'figure.autolayout': True,  # Automatically adjust layout
})

# Create example data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y1_err = 0.1 * np.random.rand(100)  # Error data

# Create figure and axis
fig, ax = plt.subplots(figsize=(8.7/2.54, 6/2.54))  # Single-column width (8.7 cm)

# Plot data
ax.plot(x, y1, label='Sin(x)', color='blue', linewidth=0.75)  # Thin lines
ax.plot(x, y2, label='Cos(x)', color='orange', linewidth=0.75)

# Add error bars
ax.errorbar(x[::10], y1[::10], yerr=y1_err[::10], fmt='o', color='blue', capsize=3, capthick=0.5, elinewidth=0.5)

# Set axis labels (lower-case, first letter capitalized)
ax.set_xlabel('Time (s)', fontsize=8)  # Units with space
ax.set_ylabel('Amplitude (m)', fontsize=8)

# Set title
ax.set_title('Example Plot for Nature Journal', fontsize=10)

# Add legend (place keys here instead of on the figure)
ax.legend(frameon=False, loc='upper right')

# Set ticks and grid
ax.tick_params(axis='both', which='major', width=0.5, size=4)
ax.tick_params(axis='both', which='minor', width=0.25, size=2)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

# Add scale bar (example for images)
# For plots, scale bars are less common, but you can use annotations if needed
ax.annotate('1 s', xy=(1, -1.2), xycoords='data', fontsize=7, ha='center', va='center')

# Save and show
plt.show()
fig.savefig('tests/nature_figure.tiff', dpi=300)