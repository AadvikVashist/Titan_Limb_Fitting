from polar_profile import analyze_complete_dataset
from sort_and_filter import sort_and_filter
from get_settings import join_strings, check_if_exists_or_write, SETTINGS
from fitting import fit_data

analyze = analyze_complete_dataset()
analyze.complete_dataset_analysis()
filter = sort_and_filter()
filter.sort_and_filter_all()
fit = fit_data()
fit.fit_all()