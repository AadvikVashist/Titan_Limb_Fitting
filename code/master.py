from get_settings import join_strings, check_if_exists_or_write, SETTINGS

from data_processing.polar_profile import analyze_complete_dataset
# from data_processing.polar_previews import preview_complete_dataset

from data_processing.process_nsa import insert_nsa
from data_processing.sort_and_filter import sort_and_filter
from data_processing.filter_using_nsa import process_nsa_data_for_fitting
from data_processing.fitting import fit_data
from data_processing.selective_analysis import select_data


from plot_generation.generate_previews import generate_cube_previews
from plot_generation.generate_four_figs import gen_quad_plots
from plot_generation.u import gen_u1_u2_figures
from plot_generation.misc_plotting import gen_plots
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
    # analyze.complete_dataset_analysis()
    
    # # print("Starting to analyze north south asymmetry data\n\n")
    # # analyze = insert_nsa()
    # # analyze.insert_nsa_data_in_all()
    # print("Starting to sort and filter data\n\n")
    # filter = sort_and_filter()
    # filter.sort_and_filter_all()
    
    # print("Starting to remove nsa data\n\n")
    # apply_nsa = process_nsa_data_for_fitting()
    # apply_nsa.select_nsa_data_in_all(emission_cutoff=25)
    

    # print("Starting to fit data\n\n")
    # fit = fit_data()
    # fit.fit_all("all", multi_process=True)
    # # fit.fit_some(0, 9, multi_process=False)

    # print("Starting to select data\n\n")
    # fit = select_data()
    # fit.check_stats()
    # fit.run_selection_on_all()
    
    plots = gen_quad_plots(devEnvironment=False)
    plots.quad_dps("C1477437155_1")
    # plots.quad_all(multi_process=True)
    
    # print("Generating previews\n\n")    
    # preview = generate_cube_previews(devEnvironment=False)
    # preview.enumerate_all(multi_process=True)
    misc = gen_plots(True)
    misc.gen_image_overlay(cube_name="C1477437155_1", band=118)

    plots = gen_u1_u2_figures(False)
    # plots.gen_u_vs_time(False)
    plots.u1_u2_all_figures(multi_process=False)
    
    