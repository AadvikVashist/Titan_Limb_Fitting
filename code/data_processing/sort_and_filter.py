from .polar_profile import analyze_complete_dataset
from get_settings import join_strings, check_if_exists_or_write, SETTINGS
import re
import time
import os
import numpy as np



class sort_and_filter:
    def __init__(self):
        self.save_dir = join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["sorted_sub_path"])
        self.data_dir = join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["analysis_sub_path"])
    def get_processed_data(self):
        all_data = {}
        if os.path.exists(join_strings(self.data_dir, SETTINGS["paths"]["cumulative_data_path"])):
            all_data = check_if_exists_or_write(join_strings(self.data_dir, SETTINGS["paths"]["cumulative_data_path"]), save=False, verbose=True)
        else:
            cubs = os.listdir(self.data_dir); cubs = [cub for cub in cubs if re.fullmatch(r'C.*_.*\.pkl', cub) is not None]
            for cub in cubs:
                cube_name = os.path.splitext(cub)[0]
                all_data[cube_name] = check_if_exists_or_write(join_strings(self.data_dir, cub), save=False, verbose=True)
        return all_data
    def sort_and_filter(self, data : dict):
        leng = len(data)
        for index, (wave_band, wave_data) in enumerate(data.items()):
            if "µm_" not in wave_band:
                continue
            for slant, slant_data in wave_data.items():
                count = np.sum(np.array(slant_data["emission_angles"]) >= 89.5)
                combined_data = zip(slant_data["pixel_indices"], slant_data["pixel_distances"], slant_data["emission_angles"], slant_data["brightness_values"])
                combined_data = sorted(combined_data, key=lambda x: x[2])
                data[wave_band][slant]["meta"]["processing"]["sorted"] = True
                if count > 0:
                    filtered_data = combined_data[:-count]
                    data[wave_band][slant]["pixel_indices"], data[wave_band][slant]["pixel_distances"], data[wave_band][slant]["emission_angles"], data[wave_band][slant]["brightness_values"] = zip(*filtered_data)
                    data[wave_band][slant]["meta"]["processing"]["filtered"] = True
                    continue
                data[wave_band][slant]["meta"]["processing"]["filtered"] = None #None means no filtering was done because there were no pixels with emission angles >= 89.5
                data[wave_band][slant]["pixel_indices"], data[wave_band][slant]["pixel_distances"], data[wave_band][slant]["emission_angles"], data[wave_band][slant]["brightness_values"] = zip(*combined_data)
            print("Finished sorting and filtering", wave_band, "| Spent", np.around(time.time() - self.cube_start_time, 3), "| expected time left:", np.around((time.time() - self.cube_start_time) * (leng - index - 1),2), end="\r")
        return data
    def sort_and_filter_all(self):
        data = self.get_processed_data()
        # for cube_name, cube_data in data.items():
        force_write = (SETTINGS["processing"]["clear_cache"] or SETTINGS["processing"]["redo_data_sorting_and_filtering"])
        appended_data = False
        cube_count = len(data)
        self.start_time = time.time()
        for index, (cube_name, cube_data) in enumerate(data.items()):        
            if os.path.exists(join_strings(self.save_dir, cube_name + ".pkl")) and not force_write:
                print("Sorted data already exists. Skipping...")
                data[cube_name] = check_if_exists_or_write(join_strings(self.save_dir, cube_name + ".pkl"), save=False)
                continue
            elif not force_write:
                appended_data = True
            self.cube_start_time = time.time()
            data[cube_name] = self.sort_and_filter(cube_data)
            check_if_exists_or_write(join_strings(self.save_dir, cube_name + ".pkl"), data=data[cube_name], save=True, force_write=True)
            time_spent = np.around(time.time() - self.cube_start_time, 3)
            percentage_completed = (index + 1) / cube_count
            total_time_left = time_spent / percentage_completed - time_spent
            print("Cube", index + 1,"of", cube_count , "| Total time for cube:", time_spent, "seconds | Total Expected time left:",
                 np.around(total_time_left,2), "seconds", "| Total time spent:", np.around(time.time() - self.start_time, 3), "seconds")        
        if (os.path.exists(join_strings(self.save_dir, SETTINGS["paths"]["cumulative_sorted_path"])) and appended_data):
            print("Sorted data already exists, but new data has been appended")
            check_if_exists_or_write(join_strings(self.save_dir, SETTINGS["paths"]["cumulative_sorted_path"]), data = data, save=True, force_write=True)
        elif force_write:
            check_if_exists_or_write(join_strings(self.save_dir, SETTINGS["paths"]["cumulative_sorted_path"]), data = data, save=True, force_write=True)
        else:
            print("Sorted not changed since last run. No changes to save...")