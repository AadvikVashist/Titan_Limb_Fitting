import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np
from scipy import stats
from scipy.stats import gaussian_kde
from pathlib import Path
import os
from settings.get_settings import SETTINGS

def setup_publication_style():
    """Set up publication-ready matplotlib style."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'stix',
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 6,
        'figure.titlesize': 9,
        'axes.grid': True,
        'axes.grid.which': 'major',
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'axes.linewidth': 0.5,
        'axes.edgecolor': 'black',
        'lines.linewidth': 0.5,
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'figure.figsize': (4.5, 2.6),
        'figure.constrained_layout.use': True
    })

class PlotColors:
    """Color scheme for plots using a colorblind-friendly palette."""
    MAIN = '#3288BD'      # Blue that's distinguishable for most color vision types
    MEAN = '#D53E4F'      # Vibrant red that maintains contrast
    MEDIAN = '#66C2A5'    # Teal that's distinct from blue and red
    DENSITY = '#F46D43'   # Orange that contrasts well with blue
    POINTS = '#5E4FA2'    # Purple that maintains visibility

def plot_parameter_distribution(
    values,
    output_dir,
    filename_base,
    x_label='Value',
    title='Parameter Distribution',
    x_min=-4.5,
    x_max=1.5,
    show_minmax=True
):
    """
    Create a publication-ready distribution plot with histogram and violin plot.
    
    Parameters
    ----------
    values : array-like
        The data values to plot
    output_dir : str or Path
        Directory to save the output files
    filename_base : str
        Base filename for saving plots (without extension)
    x_label : str
        Label for x-axis
    title : str
        Title for the plot
    x_min : float
        Minimum x-axis value
    x_max : float
        Maximum x-axis value
    show_minmax : bool
        Whether to show min/max lines on violin plot
    """
    setup_publication_style()
    
    # Create figure with adjusted height ratio
    fig, (ax1, ax2) = plt.subplots(2, 1, height_ratios=[1.6, 1])
    plt.subplots_adjust(hspace=0.25, left=0.1, right=0.95, top=0.95, bottom=0.15)
    
    # Create histogram with hatching for better visibility
    counts, bins, _ = ax1.hist(values, 
                              bins=100,
                              color=PlotColors.MAIN,
                              edgecolor='black',
                              linewidth=0.5,
                              alpha=0.7,
                              hatch='/')
    
    # Add kernel density estimate with dashed pattern
    density = gaussian_kde(values)
    xs = np.linspace(x_min, x_max, 200)
    ax1.plot(xs, density(xs) * len(values) * (bins[1] - bins[0]), 
             color=PlotColors.DENSITY, 
             linewidth=1.0,
             linestyle='--',
             alpha=0.9)
    
    # Calculate statistics
    mean_value = values.mean()
    median_value = values.median()
    std_dev = values.std()
    
    # Add mean and median lines to both plots with distinct patterns
    for ax in [ax1, ax2]:
        ax.axvline(mean_value, color=PlotColors.MEAN, linestyle='--', linewidth=1.0)
        ax.axvline(median_value, color=PlotColors.MEDIAN, linestyle=':', linewidth=1.5)
    
    # Customize histogram
    ax1.set_title(title, pad=8)
    ax1.set_ylabel('Frequency')
    
    # Set consistent x-axis for both plots
    for ax in [ax1, ax2]:
        ax.set_xlim(x_min, x_max)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis='x', which='both', direction='out')
        
        if ax == ax2:
            ticks = ax.get_xticks()
            labels = [f'{x:.1f}' for x in ticks]
            zero_index = list(ticks).index(0.0) if 0.0 in ticks else None
            if zero_index is not None:
                labels[zero_index] = r'$\mathbf{0.0}$'
            ax.set_xticklabels(labels)
    
    # Remove x-axis labels from top plot
    ax1.set_xticklabels([])
    
    # Add statistics box
    legend_text = (f'Mean: {mean_value:.2f}\n'
                  f'Median: {median_value:.2f}\n'
                  f'$\sigma$: {std_dev:.2f}')
    ax1.text(0.98, 0.90, legend_text,
             transform=ax1.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             fontsize=6,
             bbox=dict(boxstyle='round',
                      facecolor='white',
                      alpha=0.9,
                      linewidth=0.5,
                      pad=0.4))
    
    # Create violin plot with hatching for better visibility
    violin = sns.violinplot(x=values, ax=ax2, 
                   color=PlotColors.MAIN,
                   cut=0)
    
    # Add hatching to violin plot
    for patch in violin.collections:
        patch.set_hatch('/')
        patch.set_alpha(0.7)
    
    sns.stripplot(x=values, ax=ax2, 
                  color=PlotColors.POINTS,
                  alpha=0.4,
                  size=1.5,
                  jitter=0.15,
                  rasterized=True)
    
    # Customize violin plot
    ax2.set_xlabel(x_label)
    ax2.set_yticks([])
    
    # Add min/max annotations if requested
    if show_minmax:
        min_val = values.min()
        max_val = values.max()
        for val in [min_val, max_val]:
            ax2.axvline(val, color='gray', linestyle=':', linewidth=0.75, alpha=0.5)
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plots
    for ext in ['png']:
        output_path = output_dir / f'{filename_base}.{ext}'
        plt.savefig(
            output_path,
            format=ext,
            bbox_inches='tight',
            pad_inches=0.01,
            dpi=600
        )
    
    plt.close()
    plt.clf()
    return fig, (ax1, ax2)

def plot_limb_darkening_distribution(data_path=None, is_dev=True):
    """
    Create distribution plot specifically for limb darkening coefficients.
    
    Parameters
    ----------
    data_path : str or Path, optional
        Path to the CSV file containing the data. If None, uses default from settings.
    is_dev : bool, default=True
        Whether to save to development or production figures directory
    """
    if data_path is None:
        data_path = os.path.join(SETTINGS["paths"]["parent_data_path"], "Titan SRTC++ Analysis.csv")
    
    # Read and filter data
    data = pd.read_csv(data_path)
    data = data[data['Usable'] == 'yes']
    values = data['Model Output µ1+µ2 Value']
    
    # Determine output directory from settings
    base_path = SETTINGS["paths"]["parent_figures_path"]
    figures_subpath = SETTINGS["paths"]["dev_figures_sub_path"] if is_dev else SETTINGS["paths"]["prod_figures_sub_path"]
    output_dir = os.path.join(base_path, figures_subpath, "holistic_stats")
    
    return plot_parameter_distribution(
        values=values,
        output_dir=output_dir,
        filename_base='limb_darkening_distribution',
        x_label='$\mu_1$+$\mu_2$ Value',
        title='Limb Darkening Coefficient Distribution'
    ) 