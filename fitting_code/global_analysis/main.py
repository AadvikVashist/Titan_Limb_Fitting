from .analyze_trends import trend_analysis
from fitting_code.global_analysis.distribution_plots import plot_limb_darkening_distribution
def derive_all_trends(devEnvironment: bool = True):
    a = trend_analysis(devEnvironment=devEnvironment)
    
    a.stacked_u_vs_wave()
    a.plot_transitional_period()


    a.trust_mapping()

    plot_limb_darkening_distribution(is_dev=devEnvironment)

    a.save(True)
