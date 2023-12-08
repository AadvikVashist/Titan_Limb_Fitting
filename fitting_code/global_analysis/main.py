from .analyze_trends import trend_analysis
def derive_all_trends():
    a = trend_analysis()
    # a.look_at_odd_data()
    a.plot_transitional_period()
    a.trust_mapping()
    # a.three_d_plot()
    a.save(True)
    # a.three_d_plot_plotly()
