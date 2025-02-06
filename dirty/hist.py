import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy import stats

# Set up a custom style
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'] + plt.rcParams['font.serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'axes.linewidth': 0.5,
    'axes.edgecolor': 'black'
})

# Read the CSV data (update path as needed)
data = pd.read_csv('/Users/aadvik/Desktop/Work/Titan/Projects/Titan_Limb_Fitting/dirty/Titan SRTC++ Analysis - Copy of Data.csv')

# Filter out rows where 'Usable' is 'no'
data = data[data['Usable'] == 'yes']

# Calculate the full range of the data
x_min = int(data['Model Output µ1+µ2 Value'].min())
x_max = int(data['Model Output µ1+µ2 Value'].max())+1

# Create the figure and axis objects with adjusted height ratio
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])

# Create the histogram with linear scale
counts, bins, _ = ax1.hist(data['Model Output µ1+µ2 Value'], bins=100, color='#5778a4', edgecolor='black', alpha=0.7)

# Customize the histogram plot
ax1.set_title('Distribution of Model Output $\mu_1$+$\mu_2$ Values', fontweight='bold')
ax1.set_xlabel('Model Output $\mu_1$+$\mu_2$ Value')
ax1.set_ylabel('Frequency')

# Set x-axis limits for histogram
ax1.set_xlim(x_min, x_max)

# Calculate and add statistical information
mean_value = data['Model Output µ1+µ2 Value'].mean()
median_value = data['Model Output µ1+µ2 Value'].median()
std_dev = data['Model Output µ1+µ2 Value'].std()
skewness = stats.skew(data['Model Output µ1+µ2 Value'])
kurtosis = stats.kurtosis(data['Model Output µ1+µ2 Value'])

ax1.axvline(mean_value, color='#d62728', linestyle='--', linewidth=1.5, label=f'Mean: {mean_value:.2f}')
ax1.axvline(median_value, color='#2563eb', linestyle='--', linewidth=2, label=f'Median: {median_value:.2f}')

# Add legend and statistical information
ax1.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
stats_text = f'Std Dev: {std_dev:.2f}\nSkewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}'
ax1.text(0.97, 0.25, stats_text, transform=ax1.transAxes, verticalalignment='top', horizontalalignment='right', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Improve the violin plot
sns.violinplot(x=data['Model Output µ1+µ2 Value'], ax=ax2, color='#5778a4', cut=0, label='Distribution')
sns.stripplot(x=data['Model Output µ1+µ2 Value'], ax=ax2, color='black', alpha=0.4, size=2, label='Data Points')

# Add vertical lines for mean and median to violin plotπ
ax2.axvline(mean_value, color='#d62728', linestyle='--', linewidth=1.5, label=f'Mean: {mean_value:.2f}')
ax2.axvline(median_value, color='#2563eb', linestyle='--', linewidth=2, label=f'Median: {median_value:.2f}')

# Set the x-axis limits for violin plot to match the histogram
ax2.set_xlim(x_min, x_max)

ax2.set_xlabel('Model Output $\mu_1$+$\mu_2$ Value')
ax2.set_title('Violin Plot with Data Points of Model Output $\mu_1$+$\mu_2$ Values')

# Remove y-axis labels for the violin plot
ax2.set_yticks([])

# Add legend to violin plot
ax2.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

# Adjust layout and display the plot
plt.tight_layout()

# Optional: Save the plot as a high-resolution image
plt.savefig('dirty/output/titan_srtc_histogram_violin_standardized.pdf', dpi=300, bbox_inches='tight')
plt.savefig('dirty/output/titan_srtc_histogram_violin_standardized.png', dpi=300, bbox_inches='tight')

plt.show()