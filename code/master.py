from data_processing.polar_profile import analyze_complete_dataset
# from data_processing.polar_previews import preview_complete_dataset

from data_processing.sort_and_filter import sort_and_filter
from get_settings import join_strings, check_if_exists_or_write, SETTINGS
from data_processing.fitting import fit_data
from data_processing.selective_analysis import select_data
from plot_generation.generate_previews import generate_cube_previews

from plot_generation.generate_four_figs import gen_quad_plots
# preview = preview_complete_dataset()
# preview.complete_dataset_preview()

# print("Starting to analyze dataset data\n\n")
# analyze = analyze_complete_dataset()
# analyze.complete_dataaset_analysis()
# print("Starting to sort and filter data\n\n")
# filter = sort_and_filter()
# filter.sort_and_filter_all()

# print("Starting to fit data\n\n")
# fit = fit_data()
# fit.fit_all("all")

# print("Starting to select data\n\n")
# fit = select_data()
# fit.run_selection_on_all()

if __name__ == '__main__':
    # preview = preview_complete_dataset()
    # preview.complete_dataset_preview()

    # print("Starting to analyze dataset data\n\n")
    # analyze = analyze_complete_dataset()
    # analyze.complete_dataaset_analysis()
    # print("Starting to sort and filter data\n\n")
    # filter = sort_and_filter()
    # filter.sort_and_filter_all()

    # print("Starting to fit data\n\n")
    # fit = fit_data()
    # fit.fit_all("all", multi_process=True)

    # print("Starting to select data\n\n")
    # fit = select_data()
    # fit.run_selection_on_all()
    print("Generating previews\n\n")    
    preview = generate_cube_previews()
    preview.enumerate_all(multi_process=True)
    # plots = gen_quad_plots()
    # plots.quad_all(multi_process=True)