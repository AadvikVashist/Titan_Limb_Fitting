from polar_profile import analyze_complete_dataset
from polar_previews import preview_complete_dataset

from sort_and_filter import sort_and_filter
from get_settings import join_strings, check_if_exists_or_write, SETTINGS
from fitting import fit_data

preview = preview_complete_dataset()
preview.complete_dataset_preview()
analyze = analyze_complete_dataset()
analyze.complete_dataset_analysis()
filter = sort_and_filter()
filter.sort_and_filter_all()
fit = fit_data()
fit.fit_all()