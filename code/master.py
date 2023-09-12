from polar_profile import analyze_complete_dataset
from polar_previews import preview_complete_dataset

from sort_and_filter import sort_and_filter
from get_settings import join_strings, check_if_exists_or_write, SETTINGS
from fitting import fit_data

from generate_previews import generate_cube_previews
# preview = preview_complete_dataset()
# preview.complete_dataset_preview()

# print("Starting to analyze dataset data\n\n")
# # analyze = analyze_complete_dataset()
# # analyze.complete_dataset_analysis()
# print("Starting to sort and filter data\n\n")
# filter = sort_and_filter()
# filter.sort_and_filter_all()

# print("Starting to fit data\n\n")
# fit = fit_data()
# fit.fit_all("all")

print("Generating previews\n\n")
preview = generate_cube_previews()
preview.enumerate_all()