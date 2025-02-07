from .analyze_trends import trend_analysis
from fitting_code.global_analysis.distribution_plots import plot_limb_darkening_distribution
def derive_all_trends(devEnvironment: bool = True):
    plot_limb_darkening_distribution(is_dev=devEnvironment)

    a = trend_analysis(devEnvironment=devEnvironment)
    # a.look_at_odd_data()
    a.plot_transitional_period()
    a.trust_mapping()
    # a.three_d_plot()
    a.save(True)
    # a.three_d_plot_plotly()
