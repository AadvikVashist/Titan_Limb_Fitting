
from settings.get_settings import join_strings, check_if_exists_or_write, SETTINGS

from .polar_profile import analyze_complete_dataset

from .process_nsa import insert_nsa
from .sort_and_filter import sort_and_filter
from .filter_using_nsa import process_nsa_data_for_fitting
from .fitting import fit_data
from .selective_analysis import select_data
from typing import Union
import shutil

def string_fill(string: str, fill_char: str = "-", split_char: str = "> "):
    split_spaces = len(split_char) - 1
    split_char = split_char.strip()
    if split_char == ">":
        split_left = ">"
        split_right = "<"
    elif split_char == "<":
        split_left = "<"
        split_right = ">"
    elif split_char == "-":
        split_left = "-"
        split_right = "-"
    elif split_char == "|":
        split_left = "|"
        split_right = "|"
    elif split_char == "(":
        split_left = "("
        split_right = ")"
    elif split_char == ")":
        split_left = ")"
        split_right = "("
    else:
        split_left = split_char
        split_right = split_char
    split_left = split_left + split_spaces * " "
    split_right = split_spaces * " " + split_right
    
    leng = len(string)
    space_avail = (shutil.get_terminal_size().columns - leng)
    lr_leng = int(space_avail / 2) - len(split_left)
    space_avail = space_avail % 2 
    print(fill_char * lr_leng + split_left + string + split_right + fill_char * (lr_leng + space_avail))
def process_nsa_data():
    string_fill("ANALYZE NSA DATA")
    analyze = insert_nsa()
    analyze.insert_nsa_data_in_all()
    
def run_all_limb_processing(multiprocess: Union[bool, int] = False, emission_cutoff: int = 25):
    string_fill("FITTING POLAR PROFILE")
    analyze = analyze_complete_dataset()
    analyze.complete_dataset_analysis(multiprocess)

    string_fill("SORT AND FILTER 1")
    filter = sort_and_filter()
    filter.sort_and_filter_all()
    
    string_fill("FILTER USING NSA")
    apply_nsa = process_nsa_data_for_fitting()
    apply_nsa.select_nsa_data_in_all(emission_cutoff=emission_cutoff)
    
    string_fill("FITTING DATA")
    fit = fit_data()
    fit.fit_all("all", multi_process=multiprocess)

    string_fill("SELECTING FITS")
    fit = select_data()
    fit.run_selection_on_all()
    string_fill("ALL DATA PROCESSED")
