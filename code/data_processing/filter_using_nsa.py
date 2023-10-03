from .polar_profile import analyze_complete_dataset

from get_settings import join_strings, check_if_exists_or_write, SETTINGS, get_cumulative_filename
import re
import time
import os
import numpy as np
import pyvims
import matplotlib.pyplot as plt

class process_nsa_data_for_fitting:
    def __init__(self):
        self.save_dir = join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["nsa_filtered_out_sub_path"])
        self.data_dir = join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["sorted_sub_path"])
        self.nsa_dir = join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["nsa_data_sub_path"])
        cubes_location = join_strings(
                SETTINGS["paths"]["parent_data_path"],
                SETTINGS["paths"]["cube_sub_path"],)
        self.cubes_location = cubes_location
        folders = os.listdir(self.cubes_location)
        self.cubes = [
            folder
            for folder in folders
            if folder.startswith("C") and folder[-2] == ("_")
        ]
        self.cubes.sort()
    def get_nsa_data(self):
        all_data = {}
        if os.path.exists(join_strings(self.nsa_dir, get_cumulative_filename("nsa_data_sub_path"))):
            all_data = check_if_exists_or_write(join_strings(self.nsa_dir, get_cumulative_filename("nsa_data_sub_path")), save=False, verbose=True)
        else:
            cubs = os.listdir(self.nsa_dir); cubs = [cub for cub in cubs if re.fullmatch(r'C.*_.*\.pkl', cub) is not None]
            for cub in cubs:
                cube_name = os.path.splitext(cub)[0]
                all_data[cube_name] = check_if_exists_or_write(join_strings(self.nsa_dir, cub), save=False, verbose=True)
        return all_data        
    def get_sorted_data(self):
        all_data = {}
        if os.path.exists(join_strings(self.data_dir, get_cumulative_filename("sorted_sub_path"))):
            all_data = check_if_exists_or_write(join_strings(self.data_dir, get_cumulative_filename("sorted_sub_path")), save=False, verbose=True)
        else:
            cubs = os.listdir(self.data_dir); cubs = [cub for cub in cubs if re.fullmatch(r'C.*_.*\.pkl', cub) is not None]
            for cub in cubs:
                cube_name = os.path.splitext(cub)[0]
                all_data[cube_name] = check_if_exists_or_write(join_strings(self.data_dir, cub), save=False, verbose=True)
        return all_data
    
    def try_to_remove_outliers(self, data, threshold = 2):
        # Define the threshold for the width of a 'True' section

        # Initialize variables to keep track of the start and end of a 'True' section
        start = None
        end = None

        # Iterate through the array
        for i, value in enumerate(data):
            if value:
                # If this is the start of a 'True' section, record the index
                if start is None:
                    start = i
                # Update the end index to the current position
                end = i
            else:
                # If a 'True' section is less than the threshold, set it to 'False'
                if start is not None and end - start + 1 <= threshold:
                    data[start:end+1] = False
                # Reset the start and end variables for the next 'True' section
                start = None
                end = None

        # Ensure that the last 'True' section, if any, is also checked
        if start is not None and end - start + 1 <= threshold:
            data[start:end+1] = False
        return data

    def process_nsa(self, data : dict, cube_root: str, cube: str = None, force=False, nsa_data = None, emission_cutoff = 0):
        
        leng = len(data)
        thresh = 0.5
        if emission_cutoff == 0 and nsa_data is None:
            raise SyntaxError("Either emission_cutoff or nsa_data must be provided")
        elif nsa_data is not None:
            nsa_latitude = nsa_data["nsa_latitude"]

        ind_shift = 0
        count = 0
        for index, (wave_band, wave_data) in enumerate(data.items()):
            if "µm_" not in wave_band:
                ind_shift+=1
                continue
            if index >= 96+ind_shift:
                image = data["meta"]["cube_ir"]["bands"][index-ind_shift-96]
                lat = data["meta"]["cube_ir"]["lat"]
                center_point = data["meta"]["center_of_cube_ir"]; center_point = (int(center_point[0]), int(center_point[1]))
                lat_of_center= data["meta"]["cube_ir"]["lat"][center_point[0],center_point[1]]

            else:
                image = data["meta"]["cube_vis"]["bands"][index-ind_shift]
                lat = data["meta"]["cube_vis"]["lat"]
                center_point = data["meta"]["center_of_cube_vis"]; center_point = (int(center_point[0]), int(center_point[1]))
                lat_of_center = data["meta"]["cube_vis"]["lat"][center_point[0],center_point[1]]
            # image_fosho = np.where(abs(lat - nsa_latitude) > 2, image, image/2)
            for slant, slant_data in wave_data.items():
                data[wave_band][slant]["filtered"] = data[wave_band][slant].copy()
                data[wave_band][slant]["meta"]["processing"]["nsa_filter"] = False
                data[wave_band][slant]["meta"]["processing"]["emission_filter"] = False

                slant_angle = slant
                emission = np.array(slant_data["emission_angles"])
                
                if nsa_data is not None:
                    slant_lats = [lat[indx[0], indx[1]] for indx in slant_data["pixel_indices"]]
                    min_slant = np.min(slant_lats)
                    max_slant = np.max(slant_lats)
                    if nsa_latitude + thresh < min_slant or nsa_latitude - thresh > max_slant: #not in range
                        indices_mask = np.array([True for emission_angle in emission])
                        data[wave_band][slant]["meta"]["processing"]["nsa_filter"] = None
                        continue
                    elif slant == 90 or slant == 270: #parallel to nsa so should be fine
                        indices_mask = np.array([True for emission_angle in emission])
                        data[wave_band][slant]["meta"]["processing"]["nsa_filter"] = None
                        continue
                    indices_mask = (slant_lats > np.max((nsa_latitude, lat_of_center))) | (slant_lats < np.min((nsa_latitude, lat_of_center)))
                    indices_mask = self.try_to_remove_outliers(indices_mask)
                    data[wave_band][slant]["meta"]["processing"]["nsa_filter"] = True
                    # if index > 62:
                    #     pixel_indices = np.array(slant_data["pixel_indices"])
                    #     plt.imshow(image_fosho, cmap="gray")
                    #     plt.scatter(pixel_indices[:,1], pixel_indices[:,0], label=slant_angle)
                    #     plt.scatter(pixel_indices[indices_mask,1], pixel_indices[indices_mask,0], label=slant_angle)
                    #     plt.show()
                if emission_cutoff != 0 and nsa_data is None:
                    indices_mask = np.array([emission_angle > emission_cutoff for emission_angle in emission])
                    data[wave_band][slant]["meta"]["processing"]["emission_filter"] = True

                elif emission_cutoff == 0 and nsa_data is not None:
                    indices_mask = np.where(emission > emission_cutoff, indices_mask, False)
                    data[wave_band][slant]["meta"]["processing"]["emission_filter"] = True
                else: 
                    raise SyntaxError("Either emission_cutoff or nsa_data must be provided this is bad")                  

                slant_b = np.array([image[pixel_index[0], pixel_index[1]] for pixel_index in data[wave_band][slant]["pixel_indices"]])
                if not np.all(slant_b == data[wave_band][slant]["brightness_values"]):
                    raise ValueError("Brightness values not equal")
    
                data[wave_band][slant]["pixel_indices"] = data[wave_band][slant]["pixel_indices"][indices_mask]
                data[wave_band][slant]["pixel_distances"] = data[wave_band][slant]["pixel_distances"][indices_mask]
                data[wave_band][slant]["emission_angles"] = data[wave_band][slant]["emission_angles"][indices_mask]
                data[wave_band][slant]["brightness_values"] = data[wave_band][slant]["brightness_values"][indices_mask]
                slant_b = np.array([image[pixel_index[0], pixel_index[1]] for pixel_index in data[wave_band][slant]["pixel_indices"]])
                if not np.all(slant_b == data[wave_band][slant]["brightness_values"]):
                    raise ValueError("Brightness values not equal")
                count += np.count_nonzero(~indices_mask)
            # plt.scatter(data["meta"]["center_of_cube_vis"][1], data["meta"]["center_of_cube_vis"][0], marker="*", color="black", label="cube center")
            # plt.show()
            print("Finished nsa calculations", wave_band, "| Removed", count,"values | Spent", np.around(time.time() - self.cube_start_time, 3), "| expected time left:", np.around((time.time() - self.cube_start_time) * (leng - index - 1),2), end="\r")
        return data
    
    def get_stats(self, filtered_data):
        sorted_data = self.get_sorted_data()
        cube_count = 0
        image_count = 0
        slant_count = 0
        data_point_count = 0
        post_filtering_data_point_count = 0
        lists = []
        for cube, cube_data in sorted_data.items():
            cube_count += 1
            for wave_band, wave_data in cube_data.items():
                if "µm_" not in wave_band:
                    continue
                image_count += 1
                for slant, slant_data in wave_data.items():
                    slant_count += 1
                    data_point_count += len(slant_data["emission_angles"])
                    post_filtering_data_point_count += len(filtered_data[cube][wave_band][slant]["emission_angles"])
                    lists.append(len(filtered_data[cube][wave_band][slant]["emission_angles"]))
        print("Cube Count:", cube_count)
        print("Image Count:", image_count)
        print("Slant Count:", slant_count)
        print("Data Point Count:", data_point_count)
        print("Post Filtering Data Point Count:", post_filtering_data_point_count)
        print("Percentage of data points removed:", 1 - post_filtering_data_point_count / data_point_count)
        print("Average Data Points per Slant:", data_point_count / slant_count)
        print("Average is now", post_filtering_data_point_count / slant_count, "with min being", np.min(lists), "and max being", np.max(lists), "and std being", np.std(lists))


    def select_nsa_data_in_all(self, emission_cutoff = 0, nsa_data : bool = False):
        raw_data = self.get_sorted_data()
        nsa_data = self.get_nsa_data()
        # for cube_name, cube_data in data.items():
        force_write = (SETTINGS["processing"]["clear_cache"] or SETTINGS["processing"]["redo_nsa_geo_filtering"])
        appended_data = False
        cube_count = len(raw_data)
        self.start_time = time.time()
        for index, (cube_name, cube_data) in enumerate(raw_data.items()):      
            if os.path.exists(join_strings(self.save_dir, cube_name + ".pkl")) and not force_write:
                print("NSA data already exists. Skipping...")
                raw_data[cube_name] = check_if_exists_or_write(join_strings(self.save_dir, cube_name + ".pkl"), save=False)
                continue
            elif not force_write:
                appended_data = True
            
            self.cube_start_time = time.time()
            if nsa_data == True:
                raw_data[cube_name] = self.process_nsa(cube_data, cube_root=join_strings(self.cubes_location, cube_name), cube=cube_name, force=force_write, nsa_data = nsa_data[cube_name], emission_cutoff = emission_cutoff)
            else:
                raw_data[cube_name] = self.process_nsa(cube_data, cube_root=join_strings(self.cubes_location, cube_name), cube=cube_name, force=force_write, emission_cutoff = emission_cutoff)
            check_if_exists_or_write(join_strings(self.save_dir, cube_name + ".pkl"), data=raw_data[cube_name], save=True, force_write=True)
            time_spent = np.around(time.time() - self.cube_start_time, 3)
            percentage_completed = (index + 1) / cube_count
            total_time_left = time_spent / percentage_completed - time_spent
            print("Cube", index + 1,"of", cube_count , "| Total time for cube:", time_spent, "seconds | Total Expected time left:",
                 np.around(total_time_left,2), "seconds", "| Total time spent:", np.around(time.time() - self.start_time, 3), "seconds")        
        
        if (os.path.exists(join_strings(self.save_dir, get_cumulative_filename("nsa_filtered_out_sub_path"))) and appended_data):
            print("NSA data already exists, but new data has been appended")
            check_if_exists_or_write(join_strings(self.save_dir, get_cumulative_filename("nsa_filtered_out_sub_path")), data = raw_data, save=True, force_write=True)
        elif force_write:
            check_if_exists_or_write(join_strings(self.save_dir,get_cumulative_filename("nsa_filtered_out_sub_path")), data = raw_data, save=True, force_write=True)
        else:
            print("NSA data not changed since last run. No changes to save...")
        self.get_stats(raw_data)