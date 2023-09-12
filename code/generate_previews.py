from polar_profile import analyze_complete_dataset
from sort_and_filter import sort_and_filter
from get_settings import join_strings, check_if_exists_or_write, SETTINGS
import re
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
import pyvims
from fitting import fit_data
from PIL import Image


BANDS_WE_DONT_LIKE = [0, 55, 56, 57, 63, 65, 66, 67, 79, 80, 81, 82, 83, 84, 91, 92, 93, 94, 95, 96, 99, 100, 101, 106, 107, 108, 109, 110, 119, 120, 121, 122, 126, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 147, 163, 164, 165, 166, 167, 168, 169, 170, 171, 175, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 196, 198, 199, 200, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 223, 224, 225, 226, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352]

import concurrent.futures
def save_fig(fig, save_path):
    fig.savefig(save_path, dpi=150)
class generate_cube_previews:
    def __init__(self, fit_type: str = "default"):
        self.save_dir = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["preview_sub_path"])
        self.cube_path = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"])
        self.fitted_path = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["fitted_sub_path"])
    def emission_to_normalized(self, emission_angle):
        return np.cos(np.deg2rad(emission_angle))


    def get_fitted_data(self):
        all_data = {}
        if os.path.exists(join_strings(self.fitted_path, SETTINGS["paths"]["cumulative_fitted_path"])):
            all_data = check_if_exists_or_write(join_strings(
                self.fitted_path, SETTINGS["paths"]["cumulative_fitted_path"]), save=False, verbose=True)
        else:
            cubs = os.listdir(self.data_dir)
            cubs.sort()
            cubs = [cub for cub in cubs if re.fullmatch(
                r'C.*_.*\.pkl', cub) is not None]
            for cub in cubs:
                cube_name = os.path.splitext(cub)[0]
                all_data[cube_name] = check_if_exists_or_write(
                    join_strings(self.data_dir, cub), save=False, verbose=True)
        return all_data


    def fit(self, data: dict, cube_name: str = None):
        """
        C*****_1/
            0.5µm_1/
                0
                30
                45
                ...
                
        """
        leng = len(data.keys()) - len(BANDS_WE_DONT_LIKE)
        if not os.path.exists(join_strings(self.save_dir, cube_name)):    
            os.makedirs(join_strings(self.save_dir, cube_name))
        cube_vis = pyvims.VIMS(cube_name + "_vis.cub", join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube_name), channel="vis")
        cube_ir = pyvims.VIMS(cube_name + "_ir.cub",  join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube_name), channel="ir")
        vis_lat_cub = cube_vis.lat
        ir_lat_cub = cube_ir.lat
        cube_vis = [cube_vis[index] for index in range(1, 97)]
        cube_ir = [cube_ir[index] for index in range(97,353)]
        bands_done= 1
        with concurrent.futures.ThreadPoolExecutor(4) as executor:
            futures = []
            for plot_index, (wave_band, wave_data) in enumerate(data.items()):
                if os.path.exists(join_strings(self.save_dir, cube_name, wave_band + ".png")) and not SETTINGS["processing"]["redo_previews"] and not SETTINGS["processing"]["clear_cache"]:
                    continue
                if plot_index in BANDS_WE_DONT_LIKE:
                    continue
                else:
                    bands_done +=1
                fit_obj = fit_data()
                fig, axs = plt.subplots(6,3, figsize=(16,16))
                axs = axs.flatten()
                if plot_index < 96:
                    axs[0].imshow(cube_vis[plot_index], cmap="gray")
                    axs[1].imshow(vis_lat_cub, cmap="gray")
                else:
                    axs[0].imshow(cube_ir[plot_index-96], cmap="gray")
                    axs[1].imshow(ir_lat_cub, cmap="gray")
                axs[0].set_title(cube_name + " " + wave_band)
                axs[1].set_title(cube_name + " lat")
                handles = []  # To store legend handles
                for index,(slant, slant_data) in enumerate(wave_data.items()):

                    plot_index = 2+index
                    emission_angles = np.array(slant_data["emission_angles"])
                    brightness_values = np.array(slant_data["brightness_values"])
                    normalized_distances = self.emission_to_normalized(emission_angle=emission_angles)

                    plt.title(cube_name + "  " + wave_band + " " + str(slant))
                    # axs[0].title("Brightness/Emission Angle")
                    # axs[0].plot(emission_angles, brightness_values)
                    # axs[0].set_xlabel("Emission Angle")
                    # axs[0].set_ylabel("Brightness Value")
                    
                    axs[plot_index].set_title("Brightness/Normalized")
                    
                    data_plot = axs[plot_index].plot(normalized_distances, brightness_values, label = "data")
                    # smooth_plot = axs[plot_index].plot(normalized_distances, gaussian_filter(brightness_values, sigma= 3), label = "data with gaussian")

                    axs[plot_index].set_xlabel("Normalized Distance")
                    axs[plot_index].set_ylabel("Brightness Value")
                    if index == 0:
                        handles.extend(data_plot)
                        
                        # handles.extend(smooth_plot)
                    if slant_data["meta"]["processing"]["fitted"] == True:
                        fitted_values = fit_obj.quadratic_limb_darkening(normalized_distances,*list(slant_data["fit"]["quadratic"]["optimal_fit"]["fit_params"].values())) 
                        fit_plot = axs[plot_index].plot(normalized_distances, fitted_values, label = "fit")
                        if len(handles) == 2:
                            handles.extend(fit_plot)
                            
                    # plt.plot(distances, self.limb_darkening_function(
                        # distances, popt[0], popt[1]))
                    # Smooth data using moving average
                fig.tight_layout()
                if len(handles) == 3:
                    fig.legend(handles, ["data", "smoothed data", "fit"], loc='upper left')  # Set the legend to the top-left corner
                else:
                    fig.legend(handles, ["data", "smoothed data"], loc='upper left')  # Set the legend to the top-left corner
                # List to keep track of the futures
                fig.savefig(join_strings(self.save_dir, cube_name, wave_band + ".png"), dpi=150)
                plt.close("all")  # Close the current figure
                plt.clf()
                future = executor.submit(save_fig, join_strings(self.save_dir, cube_name, wave_band + ".png"))
                futures.append(future)
                time_spent = np.around(time.time() - self.cube_start_time, 3)
                percentage_completed = (bands_done) / leng
                total_time_left = time_spent / percentage_completed - time_spent

                print("Finished fitting", wave_band, "| Spent", time_spent, "so far | expected time left:",
                    np.around(total_time_left, 2), "| total time for cube :",  np.around(total_time_left + time_spent,3), end="\r")
            concurrent.futures.wait(futures)
        

    def enumerate_all(self):
        data = self.get_fitted_data()
        force_write = (SETTINGS["processing"]["clear_cache"]
                       or SETTINGS["processing"]["redo_previews"])
        appended_data = False
        cube_count = len(data)
        self.start_time = time.time()

        for index, (cube_name, cube_data) in enumerate(data.items()):
            if os.path.exists(join_strings(self.save_dir, cube_name + ".pkl")) and not force_write:
                try:
                    data[cube_name] = check_if_exists_or_write(
                        join_strings(self.save_dir, cube_name + ".pkl"), save=False)
                    print("fitted data already exists. Skipping...")
                    continue
                except:
                    print("fitted data corrupted. Processing...")
            elif not force_write:
                appended_data = True
            self.cube_start_time = time.time()
            # only important line in this function
            self.fit(cube_data, cube_name)

            time_spent = np.around(time.time() - self.cube_start_time, 3)
            percentage_completed = (index + 1) / cube_count
            total_time_left = time_spent / percentage_completed - time_spent
            print("Cube", index + 1,"of", cube_count , "| Total time for cube:", time_spent, "seconds | Total Expected time left:",
                 np.around(total_time_left,2), "seconds", "| Total time spent:", np.around(time.time() - self.start_time, 3), "seconds")        

