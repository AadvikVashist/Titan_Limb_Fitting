from .polar_profile import analyze_complete_dataset
from settings.get_settings import join_strings, check_if_exists_or_write, SETTINGS, get_cumulative_filename
import re
import time
import os
import numpy as np
import matplotlib.pyplot as plt


class sort_and_filter:
    def __init__(self):
        self.save_dir = join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["sorted_sub_path"])
        self.data_dir = join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["analysis_sub_path"])
    def get_processed_data(self):
        all_data = {}
        if os.path.exists(join_strings(self.data_dir, get_cumulative_filename("analysis_sub_path"))):
            all_data = check_if_exists_or_write(join_strings(self.data_dir, get_cumulative_filename("analysis_sub_path")), save=False, verbose=True)
        else:
            cubs = os.listdir(self.data_dir); cubs = [cub for cub in cubs if re.fullmatch(r'C.*_.*\.pkl', cub) is not None]
            for cub in cubs:
                cube_name = os.path.splitext(cub)[0]
                all_data[cube_name] = check_if_exists_or_write(join_strings(self.data_dir, cub), save=False, verbose=True)
        return all_data
    def sort_and_filter(self, data : dict):
        leng = len(data)
        ind_shift = 0
        for index, (wave_band, wave_data) in enumerate(data.items()):
            if "µm_" not in wave_band:
                ind_shift+=1
                continue
            if index >= 96+ind_shift:
                eme = data["meta"]["cube_ir"]["eme"]
                image = data["meta"]["cube_ir"]["bands"][index-ind_shift-96]
            else:
                eme = data["meta"]["cube_vis"]["eme"]
                image = data["meta"]["cube_vis"]["bands"][index-ind_shift]

            # if index > 96:
            #     mask = np.where(data["meta"]["cube_ir"]["ground"],data["meta"]["cube_ir"]["lat"] , 0)
            # else:
            #     mask = np.where(data["meta"]["cube_ir"]["ground"], data["meta"]["cube_ir"]["lat"], 0)
                
            # # pixel indices are ACTUAL INDEXES (though they are plotted opposite to plt)
            # for slant, slant_data in wave_data.items():
            #     for pixel_index in slant_data["pixel_indices"]:
            #         mask[pixel_index[0], pixel_index[1]] = 10    
            #     plt.imshow(mask)
            #     plt.show()

            # plt.imshow(data["meta"]["cube_ir"]["lat"])
            # plt.show()
            # plt.imshow(mask)
            for slant, slant_data in wave_data.items():
                
                # count = np.sum(np.array(slant_data["emission_angles"]) >= 89.5)
                # plt.scatter(np.array(data[wave_band][slant]["pixel_indices"])[:,1], np.array(data[wave_band][slant]["pixel_indices"])[:,0])

                # filtered_data= [index for index in slant_data["pixel_indices"] if mask[index[0], index[1]] == 1] #ensure that the shit is on the ground, but also not some ass emission angle
                data[wave_band][slant]["polar_profile"] = data[wave_band][slant]
                combined_data = zip(slant_data["pixel_indices"], slant_data["pixel_distances"], slant_data["emission_angles"], slant_data["brightness_values"])
                
                combined_data = sorted(combined_data, key=lambda x: x[2])
                data[wave_band][slant]["meta"]["processing"]["sorted"] = True
                
                pixel_indices, pixel_distances, eme_angles, b_values = zip(*combined_data)
                pixel_indices, pixel_distances, eme_angles, b_values = np.array(pixel_indices), np.array(pixel_distances), np.array(eme_angles), np.array(b_values)
                # plt.scatter(pixel_indices[:,1], pixel_indices[:,0], marker= "*") #test baseline
                slant_b = np.array([image[pixel_index[0], pixel_index[1]] for pixel_index in pixel_indices])
                if not np.all(slant_b == b_values):
                    raise ValueError("Brightness values not equal")
                data[wave_band][slant]["sorted"] = data[wave_band][slant]
                data[wave_band][slant]["sorted"]["pixel_indices"] = pixel_indices; data[wave_band][slant]["sorted"]["pixel_distances"] = pixel_distances; data[wave_band][slant]["sorted"]["emission_angles"] = eme_angles; data[wave_band][slant]["sorted"]["brightness_values"] = b_values
                filter_data_by_eme_and_ground_arr = np.where(eme_angles <= 89, True, False)
                # filter_data_by_eme_and_ground_arr = np.where(mask[pixel_indices[:,0], pixel_indices[:,1]] == True, filter_data_by_eme_and_ground_arr, False)
                
                if np.min(np.diff(eme_angles[filter_data_by_eme_and_ground_arr])) < -1:
                    plt.imshow(eme, cmap="gray")
                    plt.scatter(pixel_indices[:,1], pixel_indices[:,0], c = range(pixel_indices.shape[0]), cmap = plt.get_cmap("hsv"))
                    eme_angle = [eme[pix[0], pix[1]] for pix in pixel_indices]
                    plt.show() 
                    raise ValueError("pixel_distances or eme angles not monotonically increasing")
                if (not any(filter_data_by_eme_and_ground_arr)) or len(filter_data_by_eme_and_ground_arr) == 0:
                    # plt.imshow(np.where(mask, eme, 0))
                    plt.scatter(pixel_indices[:,1], pixel_indices[:,0]) #test baseline
                    filter_data_by_eme_and_ground_arr = np.where(eme_angles <= 89.5, True, False)
                    plt.scatter(pixel_indices[filter_data_by_eme_and_ground_arr,1], pixel_indices[filter_data_by_eme_and_ground_arr,0], marker= ".") #test baseline

                    # filter_data_by_eme_and_ground_arr = np.where(mask[pixel_indices[:,0], pixel_indices[:,1]] == True, True, False)
                    plt.scatter(pixel_indices[filter_data_by_eme_and_ground_arr,1], pixel_indices[filter_data_by_eme_and_ground_arr,0],marker="*") #test baseline

                    plt.show()
                elif not all(filter_data_by_eme_and_ground_arr):
                    pixel_indices = pixel_indices[filter_data_by_eme_and_ground_arr]
                    pixel_distances = pixel_distances[filter_data_by_eme_and_ground_arr]
                    eme_angles = eme_angles[filter_data_by_eme_and_ground_arr]
                    b_values = b_values[filter_data_by_eme_and_ground_arr]
                    data[wave_band][slant]["meta"]["processing"]["filtered"] = True
                    data[wave_band][slant]["pixel_indices"], data[wave_band][slant]["pixel_distances"], data[wave_band][slant]["emission_angles"], data[wave_band][slant]["brightness_values"] = pixel_indices, pixel_distances, eme_angles, b_values
                    # plt.scatter(np.array(data[wave_band][slant]["pixel_indices"])[:,1], np.array(data[wave_band][slant]["pixel_indices"])[:,0]) #check if it worked
                    continue

                data[wave_band][slant]["meta"]["processing"]["filtered"] = None #None means no filtering was done because there were no pixels with emission angles >= 89.5
                data[wave_band][slant]["pixel_indices"], data[wave_band][slant]["pixel_distances"], data[wave_band][slant]["emission_angles"], data[wave_band][slant]["brightness_values"] = pixel_indices, pixel_distances, eme_angles, b_values

            # plt.show()
            print("Finished sorting and filtering", wave_band, "| Spent", np.around(time.time() - self.cube_start_time, 3), "| expected time left:", np.around((time.time() - self.cube_start_time) * (leng - index - 1),2), end="\r")
        return data
    def get_stats(self, sorted_data):
        data = self.get_processed_data()
        cube_count = 0
        image_count = 0
        slant_count = 0
        data_point_count = 0
        post_filtering_data_point_count = 0
        lists = []
        for cube, cube_data in data.items():
            cube_count += 1
            for wave_band, wave_data in cube_data.items():
                if "µm_" not in wave_band:
                    continue
                image_count += 1
                for slant, slant_data in wave_data.items():
                    slant_count += 1
                    data_point_count += len(slant_data["emission_angles"])
                    post_filtering_data_point_count += len(sorted_data[cube][wave_band][slant]["emission_angles"])
                    lists.append(len(sorted_data[cube][wave_band][slant]["emission_angles"]))
        print("Cube Count:", cube_count)
        print("Image Count:", image_count)
        print("Slant Count:", slant_count)
        print("Data Point Count:", data_point_count)
        print("Post Filtering Data Point Count:", post_filtering_data_point_count)
        print("Percentage of data points removed:", 1 - post_filtering_data_point_count / data_point_count)
        print("Average Data Points per Slant:", data_point_count / slant_count)
        print("Average is now", post_filtering_data_point_count / slant_count, "with min being", np.min(lists), "and max being", np.max(lists), "and std being", np.std(lists))

    def sort_and_filter_all(self):
        force_write = (SETTINGS["processing"]["clear_cache"] or SETTINGS["processing"]["redo_data_sorting_and_filtering"])

        if all([cub in os.listdir(self.save_dir) or cub == get_cumulative_filename("analysis_sub_path") for cub in os.listdir(self.data_dir)]) and os.path.exists(join_strings(self.save_dir, get_cumulative_filename("sorted_sub_path"))) and not force_write:
            print("Data already sorted")
            return
        data = self.get_processed_data()
        # for cube_name, cube_data in data.items():
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
        if (os.path.exists(join_strings(self.save_dir, get_cumulative_filename("sorted_sub_path"))) and appended_data):
            print("Sorted data already exists, but new data has been appended")
            check_if_exists_or_write(join_strings(self.save_dir, get_cumulative_filename("sorted_sub_path")), data = data, save=True, force_write=True)
        elif force_write:
            check_if_exists_or_write(join_strings(self.save_dir, get_cumulative_filename("sorted_sub_path")), data = data, save=True, force_write=True)
        elif not os.path.exists(join_strings(self.save_dir, get_cumulative_filename("sorted_sub_path"))):
            print("Sorted data does not exist. Creating...")
            check_if_exists_or_write(join_strings(self.save_dir, get_cumulative_filename("sorted_sub_path")), data = data, save=True, force_write=True)
        else:
            print("Sorted not changed since last run. No changes to save...")
        self.get_stats(data)