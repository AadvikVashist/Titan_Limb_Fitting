
from settings.get_settings import join_strings, check_if_exists_or_write, SETTINGS, get_cumulative_filename
import re
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvims
from ..data_processing.fitting import fit_data

import multiprocessing
from typing import Union

def save_fig(fig, save_path):
    fig.savefig(save_path, dpi=150)

class generate_cube_previews:
    def __init__(self, devEnvironment: bool = True):
        if devEnvironment == True:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["dev_figures_sub_path"],  SETTINGS["paths"]["figure_subpath"]["preview"])
        else:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["prod_figures_sub_path"],  SETTINGS["paths"]["figure_subpath"]["preview"])
        self.devEnvironment = devEnvironment
        self.fitted_path = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["fitted_sub_path"])
    def emission_to_normalized(self, emission_angle):
        return np.cos(np.deg2rad(emission_angle))

    def get_fitted_data(self):
        all_data = {}
        if os.path.exists(join_strings(self.fitted_path,get_cumulative_filename("fitted_sub_path"))):
            all_data = check_if_exists_or_write(join_strings(
                self.fitted_path, get_cumulative_filename("fitted_sub_path")), save=False, verbose=True)
        else:
            cubs = os.listdir(self.fitted_path)
            cubs.sort()
            cubs = [cub for cub in cubs if re.fullmatch(
                r'C.*_.*\.pkl', cub) is not None]
            for cub in cubs:
                cube_name = os.path.splitext(cub)[0]
                all_data[cube_name] = check_if_exists_or_write(
                    join_strings(self.fitted_path, cub), save=False, verbose=True)
        return all_data

        x = 0
    def generate_cube_previews(self, data: dict, cube_index: int, cube_name: str = None):
        """
        C*****_1/
            0.5µm_1/
                0
                30
                45
                ...
                
        """
        leng = len(data.keys()) - len(SETTINGS["figure_generation"]["unusable_bands"])
        if not os.path.exists(join_strings(self.save_dir, cube_name)):    
            os.makedirs(join_strings(self.save_dir, cube_name))

        cube_vis = [data["meta"]["cube_vis"]["bands"][index] for index in range(0, 96)]
        cube_ir = [data["meta"]["cube_ir"]["bands"][index] for index in range(0,352-96)]
        bands_done= 1
        futures = []
        mpl.rcParams['path.simplify_threshold'] = 1.0
        mpl.style.use('fast')
        plt.rcParams['font.family'] = 'serif'
        for plot_index, (wave_band, wave_data) in enumerate(data.items()):
            if "µm_" not in wave_band:
                continue
            if plot_index in SETTINGS["figure_generation"]["unusable_bands"]:
                continue
            else:
                bands_done +=1
            band_wave = wave_band.split("_")[1] + "_" + wave_band.split("_")[0]
            if os.path.exists(join_strings(self.save_dir, cube_name, band_wave + ".png")) and not self.force_write:
                continue

            fit_obj = fit_data()
            fig, axs = plt.subplots(6,3, figsize=(16,16))
            axs = axs.flatten()
            handles = []  # To store legend handles

            if plot_index < 96:
                axs[0].imshow(cube_vis[plot_index], cmap="gray")
                axs[1].imshow(data["meta"]["cube_vis"]["lat"], cmap="gray")
                handles.extend(axs[1].plot([data["meta"]["center_of_cube_vis"][1], data["meta"]["center_of_cube_vis"][1] + np.sin(np.radians(data["meta"]["north_orientation_vis"])) * 30],
                            [data["meta"]["center_of_cube_vis"][0], data["meta"]["center_of_cube_vis"][0] - np.cos(np.radians(data["meta"]["north_orientation_vis"])) * 30], c=(0,1,0), label="North"))
            else:
                axs[0].imshow(cube_ir[plot_index-96], cmap="gray")
                axs[1].imshow(data["meta"]["cube_ir"]["lat"], cmap="gray")
                handles.extend(axs[1].plot([data["meta"]["center_of_cube_ir"][1], data["meta"]["center_of_cube_ir"][1] + np.sin(np.radians(data["meta"]["north_orientation_ir"])) * 30],
                            [data["meta"]["center_of_cube_ir"][0], data["meta"]["center_of_cube_ir"][0] - np.cos(np.radians(data["meta"]["north_orientation_ir"])) * 30], c=(0,1,0), label="North"))
            axs[0].set_title(cube_name + " " + wave_band)
            axs[1].set_title(cube_name + " lat")
            y_box =[]
            
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
                
                axs[plot_index].set_title(str(slant) + "° relative to north")
                
                data_plot = axs[plot_index].plot(normalized_distances, brightness_values, c = (1,0,0),label = "data")
                # smooth_plot = axs[plot_index].plot(normalized_distances, gaussian_filter(brightness_values, sigma= 3), label = "data with gaussian")

                axs[plot_index].set_xlabel("Normalized Distance")
                axs[plot_index].set_ylabel("Brightness Value")
                axs[plot_index].set_xlim(0,1)
                axs[plot_index].set_xticks(np.arange(0,1.01,0.05), minor=True)
                axs[plot_index].set_xticks(np.arange(0,1.01,0.1), minor=False)

                y_box.extend(brightness_values)
                if index == 0:
                    handles.extend(data_plot)
                    
                    # handles.extend(smooth_plot)
                if slant_data["meta"]["processing"]["fitted"] == True and len(slant_data["fit"]["quadratic"]["optimal_fit"]) > 0 and slant_data["fit"]["quadratic"]["optimal_fit"]["fit_params"] is not None:
                    fitted_values = fit_obj.quadratic_limb_darkening(normalized_distances,*list(slant_data["fit"]["quadratic"]["optimal_fit"]["fit_params"].values())) 
                    fit_plot = axs[plot_index].plot(normalized_distances, fitted_values, c = (0.3,0.3,0.8), label = "fit")
                    if len(handles) < 3:
                        handles.extend(fit_plot)
                
                # plt.plot(distances, self.limb_darkening_function(
                    # distances, popt[0], popt[1]))
                # Smooth data using moving average
            
            
            #set y_lim
            # min_box = np.min(y_box); max_box = np.max(y_box); box_range = max_box - min_box
            # min_box -= box_range * 0.1; max_box += box_range * 0.1

            y_range = np.max(y_box) - np.min(y_box)
            y_min = np.min(y_box) - y_range * 0.1; y_min -= y_min % 0.01
            y_max = np.max(y_box) + y_range * 0.1; y_max += y_max % 0.01

            for index in range(len(wave_data.keys())):
                plot_index = 2+index 
                axs[plot_index].grid(True, which='both', axis='both', linestyle='--')

                axs[plot_index].set_ylim(y_min,y_max)
                if y_range < 0.1:
                    axs[plot_index].set_yticks(np.arange(y_min,y_max+0.001,0.01))
                    axs[plot_index].set_yticks(np.arange(y_min,y_max+0.001,0.005), minor=True)
                elif y_range < 0.2:
                    axs[plot_index].set_yticks(np.arange(y_min,y_max+0.001,0.02))
                    axs[plot_index].set_yticks(np.arange(y_min,y_max+0.001,0.01), minor=True)

                elif y_range < 0.5:
                    axs[plot_index].set_yticks(np.arange(y_min,y_max+0.001,0.05))
                    axs[plot_index].set_yticks(np.arange(y_min,y_max+0.001,0.025))

            fig.tight_layout()
            
            
            
            if len(handles) == 3:
                fig.legend(handles, ["North", "Data", "Fit"], loc='upper left', fontsize = 15)  # Set the legend to the top-left corner
            else:
                fig.legend(handles, ["North", "Data"], loc='upper left', fontsize = 15 )  # Set the legend to the top-left corner
            # List to keep track of the futures
            fig.savefig(join_strings(self.save_dir, cube_name, band_wave + ".png"), dpi=150)
            plt.close()
            time_spent = np.around(time.time() - self.cube_start_time, 3)
            percentage_completed = (bands_done) / leng
            total_time_left = time_spent / percentage_completed - time_spent

            print("Finished previews for ", wave_band, "| Spent", time_spent, "so far | expected time left:",
                np.around(total_time_left, 2), "| total time for cube :",  np.around(total_time_left + time_spent,3), end="\r")
        
        time_spent = np.around(time.time() - self.cube_start_time, 3)
        percentage_completed = (cube_index + 1) / self.cube_count
        total_time_left = time_spent / percentage_completed - time_spent
        print("Cube", cube_index + 1,"of", self.cube_count , "| Total time for cube:", time_spent, "seconds | Total Expected time left:",
        np.around(total_time_left,2), "seconds", "| Total time spent:", np.around(time.time() - self.start_time, 3), "seconds")        

    
    def enumerate_all(self, multi_process: Union[bool,int] = False):
        data = self.get_fitted_data()
        self.force_write = (SETTINGS["processing"]["clear_cache"]
                       or SETTINGS["processing"]["redo_preview_figures_generation"])
        self.cube_count = len(data)
        self.start_time = time.time()
        if type(multi_process) == int:
            if multi_process == 1:
                multi_process = False
                multi_process_core_count = 1
            else:
                multi_process_core_count = multi_process
                multi_process = True
                args = []
        elif type(multi_process) == bool:
            if multi_process == True:
                multi_process_core_count = 3
                args =[]
            else:
                multi_process_core_count = 1
        else:
            raise ValueError("multiprocess is wrong, type needs to be bool or int")
        
        for index, (cube_name, cube_data) in enumerate(data.items()):

            self.cube_start_time = time.time()
            # only important line in this function
            if multi_process == False:
                self.generate_cube_previews(cube_data, index, cube_name)
            else:
                args.append([cube_data, index, cube_name])
        # self.generate_cube_previews(cube_data, cube_name)
        if multi_process == True:
            with multiprocessing.Pool(processes=3) as pool:
                pool.starmap(self.generate_cube_previews, args)
        # return args, self.generate_cube_previews
