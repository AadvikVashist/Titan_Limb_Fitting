
from settings.get_settings import join_strings, check_if_exists_or_write, SETTINGS

from .polar_profile import analyze_complete_dataset

from .process_nsa import insert_nsa
from .sort_and_filter import sort_and_filter
from .filter_using_nsa import process_nsa_data_for_fitting
from .fitting import fit_data
from .selective_analysis import select_data
from typing import Union

def process_nsa_data():
    print("Starting to analyze north south asymmetry data\n\n")
    analyze = insert_nsa()
    analyze.insert_nsa_data_in_all()
    
def run_all_limb_processing(multiprocess: Union[bool, int] = False, emission_cutoff: int = 25):
    
    print("Starting to analyze dataset data\n\n")
    analyze = analyze_complete_dataset()
    analyze.complete_dataset_analysis(multiprocess)

    print("Starting to sort and filter data\n\n")
    filter = sort_and_filter()
    filter.sort_and_filter_all()
    
    print("Starting to remove nsa data\n\n")
    apply_nsa = process_nsa_data_for_fitting()
    apply_nsa.select_nsa_data_in_all(emission_cutoff=emission_cutoff)
    
    print("Starting to fit data\n\n")
    fit = fit_data()
    fit.fit_all("all", multi_process=multiprocess)

    print("Starting to select data\n\n")
    fit = select_data()
    fit.check_stats()
    fit.run_selection_on_all()
    
