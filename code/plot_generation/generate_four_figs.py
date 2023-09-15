from data_processing.polar_profile import analyze_complete_dataset
from data_processing.sort_and_filter import sort_and_filter
from get_settings import join_strings, check_if_exists_or_write, SETTINGS
import re
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvims
from data_processing.fitting import fit_data

import multiprocessing


def save_fig(fig, save_path):
    fig.savefig(save_path, dpi=150)


class gen_quad_plots:
    def __init__(self, devEnvironment: bool = True):
        if devEnvironment == True:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["dev_figures_sub_path"],  SETTINGS["paths"]["quad_figure_subpath"])
        else:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["prod_figures_sub_path"],  SETTINGS["paths"]["quad_figure_subpath"])
        self.devEnvironment = devEnvironment
        # self.save_dir = join_strings(
        #     SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["plot_sub_path"])
        # self.cube_path = join_strings(
        #     SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"])
        self.selective_fitted_path = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["selected_sub_path"])

    def emission_to_normalized(self, emission_angle):
        return np.cos(np.deg2rad(emission_angle))

    def get_data(self):
        all_data = {}
        if os.path.exists(join_strings(self.selective_fitted_path, SETTINGS["paths"]["cumulative_selected_path"])):
            all_data = check_if_exists_or_write(join_strings(
                self.selective_fitted_path, SETTINGS["paths"]["cumulative_selected_path"]), save=False, verbose=True)
        else:
            cubs = os.listdir(self.selective_fitted_path)
            cubs.sort()
            cubs = [cub for cub in cubs if re.fullmatch(
                r'C.*_.*\.pkl', cub) is not None]
            for cub in cubs:
                cube_name = os.path.splitext(cub)[0]
                all_data[cube_name] = check_if_exists_or_write(
                    join_strings(self.selective_fitted_path, cub), save=False, verbose=True)
        return all_data

        x = 0

    def gen_cube_quads(self, data: dict, cube_index: int, cube_name: str = None):
        """
        C*****_1/
            0.5µm_1/
                0
                30
                45
                ...

        """
        leng = len(data.keys()) - \
            len(SETTINGS["figure_generation"]["unusable_bands"])
        if not os.path.exists(join_strings(self.save_dir, cube_name)):
            os.makedirs(join_strings(self.save_dir, cube_name))
        # cube_vis = pyvims.VIMS(cube_name + "_vis.cub", join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube_name), channel="vis")
        # cube_ir = pyvims.VIMS(cube_name + "_ir.cub",  join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube_name), channel="ir")

        cube_vis = [data["meta"]["cube_vis"]["bands"][index]
                    for index in range(0, 96)]
        cube_ir = [data["meta"]["cube_ir"]["bands"][index]
                   for index in range(0, 352-96)]
        bands_done = 1
        futures = []
        mpl.rcParams['path.simplify_threshold'] = 1.0
        mpl.style.use('fast')
        fit_obj = fit_data()

        for plot_index, (wave_band, wave_data) in enumerate(data.items()):
            if "µm_" not in wave_band:
                continue
            if plot_index in SETTINGS["figure_generation"]["unusable_bands"]:
                continue
            else:
                bands_done += 1
            band_wave = wave_band.split("_")[1] + "_" + wave_band.split("_")[0]
            if os.path.exists(join_strings(self.save_dir, cube_name, band_wave + ".png")) and not self.force_write:
                continue

            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            axs = axs.flatten()
            
            #plot the cube and the lat
            if plot_index < 96:
                suffix = "_vis"
                axs[0].imshow(cube_vis[plot_index], cmap="gray")
                axs[1].imshow(data["meta"]["cube_vis"]["lat"], cmap="gray")
            else:
                suffix = "_ir"
                axs[0].imshow(cube_ir[plot_index-96], cmap="gray")
                axs[1].imshow(data["meta"]["cube_ir"]["lat"], cmap="gray")
            shape = data["meta"]["cube_ir"]["lat"].shape
            axs[1].plot([data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][1] + np.sin(np.radians(data["meta"]["north_orientation" + suffix])) * 50],
                        [data["meta"]["center_of_cube" + suffix][0], data["meta"]["center_of_cube" + suffix][0] - np.cos(np.radians(data["meta"]["north_orientation" + suffix])) * 50], color=(0,0,0), label="North")
            
            axs[1].plot([data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][1] - np.sin(np.radians(data["meta"]["north_orientation" + suffix])) * 50],
                        [data["meta"]["center_of_cube" + suffix][0], data["meta"]["center_of_cube" + suffix][0] + np.cos(np.radians(data["meta"]["north_orientation" + suffix])) * 50], color=(0.5,0.5,0.5), label="South")            

            axs[1].scatter(data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][0], color=(1,0,0), s = 60 , label="Center of Disk")
            axs[1].scatter(data["meta"]["lowest_inc_location" + suffix][1], data["meta"]["lowest_inc_location" + suffix][0], color=(0.2,0.5,0.7), s = 60 , label="Lowest Inc")

            axs[0].set_title(cube_name + " " + wave_band)
            axs[1].set_title(cube_name + " lat")
            axs[1].set_xlim(0, shape[1]-1)
            axs[1].set_ylim(shape[0]-1, 0)
            
            #get the slant data
            north_slant = wave_data["north_side"]
            south_slant = wave_data["south_side"]
            
            #put the slant data on to the lat array 
            area = len([dat for dat in data["meta"]["cube_vis"]["ground"].flatten() if dat != False])
            length = np.sqrt(area / np.pi) 

            axs[1].plot([data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][1] + np.sin(np.radians(data["meta"]["north_orientation" + suffix] + north_slant["angle"])) * length],
                        [data["meta"]["center_of_cube" + suffix][0], data["meta"]["center_of_cube" + suffix][0] - np.cos(np.radians(data["meta"]["north_orientation" + suffix] + north_slant["angle"])) * length], color=(0.2,0.6,0.3), label="north_side_profile")  
            axs[1].plot([data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][1] + np.sin(np.radians(data["meta"]["north_orientation" + suffix] + south_slant["angle"])) * length],
                        [data["meta"]["center_of_cube" + suffix][0], data["meta"]["center_of_cube" + suffix][0] - np.cos(np.radians(data["meta"]["north_orientation" + suffix] + south_slant["angle"])) * length], color=(0.2,0.5,0.6), label="south_side_profile")  
            axs[1].legend(loc="upper left")



            #start bottom row of plots
            axs[2].set_title(cube_name + "  " + wave_band + " " + str(north_slant["angle"]) + "° relative to north")
            axs[3].set_title(cube_name + "  " + wave_band + " " + str(south_slant["angle"]) + "° relative to north")


            #figure out what the best y range is for the plot
            range_vals = list(north_slant["brightness_values"]); range_vals.extend(south_slant["brightness_values"])
            y_range = np.max(range_vals) - np.min(range_vals)
            y_min = np.min(range_vals) - y_range * 0.1; y_min -= y_min % 0.01
            y_max = np.max(range_vals) + y_range * 0.1; y_max += y_max % 0.01
            
            if y_range < 0.05:
                axs[2].set_yticks(np.arange(y_min,y_max+0.001,0.0025), minor=True)
                axs[3].set_yticks(np.arange(y_min,y_max+0.001,0.0025), minor=True)
                
            #set plot styles
            
            axs[2].set_xlabel("Normalized Distance")
            axs[2].set_ylabel("Brightness Value")
            axs[2].set_ylim(y_min,y_max)
            axs[2].set_xlim(0,1)
            axs[2].set_yticks(np.arange(y_min,y_max+0.001,0.01))
            axs[2].set_xticks(np.arange(0,1.1,0.1))
            axs[2].set_xticks(np.arange(0,1.1,0.05), minor=True)
            
            axs[3].set_xlabel("Normalized Distance")
            axs[3].set_ylabel("Brightness Value")
            axs[3].set_ylim(y_min,y_max)

            axs[3].set_xlim(0,1)
            axs[3].set_yticks(np.arange(y_min,y_max+0.001,0.01))
            axs[3].set_xticks(np.arange(0,1.1,0.1))
            axs[3].set_xticks(np.arange(0,1.1,0.05), minor=True)
            



            #plot the data
            normalized_distances = self.emission_to_normalized(emission_angle=north_slant["emission_angles"])
            axs[2].plot(normalized_distances, north_slant["brightness_values"], color= (0.2,0.6,0.3), label="north_side_profile")

            normalized_distances = self.emission_to_normalized(emission_angle=south_slant["emission_angles"])
            axs[3].plot(normalized_distances, south_slant["brightness_values"], color=(0.2,0.5,0.6), label="south_side_profile")
            

            #work with fits
            if north_slant["meta"]["processing"]["fitted"] == True:
                normalized_distances = self.emission_to_normalized(emission_angle=north_slant["emission_angles"])
                fitted_values = fit_obj.quadratic_limb_darkening(normalized_distances,*list(north_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].values()))
                axs[2].plot(normalized_distances, fitted_values, color= (0.9,0.2,0.2), label = "fit")
                string = '\n'.join([str(key) + " : " + str(item) for key,item in north_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].items()])  + "\n r2_score: " + str(north_slant["fit"]["quadratic"]["optimal_fit"]["r2"])
                axs[2].text(0.95, 0.03, string,
                        verticalalignment='bottom', horizontalalignment='right',
                        transform=axs[2].transAxes,
                        color='black', fontsize=10,
                        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
            

            
            if south_slant["meta"]["processing"]["fitted"] == True:
                normalized_distances = self.emission_to_normalized(emission_angle=south_slant["emission_angles"])
                fitted_values = fit_obj.quadratic_limb_darkening(normalized_distances,*list(south_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].values()))
                axs[3].plot(normalized_distances, fitted_values,color= (0.9,0.2,0.2), label = "fit")
                string = '\n'.join([str(key) + " : " + str(item) for key,item in south_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].items()])  + "\n r2_score: " + str(south_slant["fit"]["quadratic"]["optimal_fit"]["r2"])
                axs[3].text(0.95, 0.03, string,
                        verticalalignment='bottom', horizontalalignment='right',
                        transform=axs[3].transAxes,
                        color='black', fontsize=10,
                        bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})
            

            axs[2].legend()
            axs[2].grid(True, which='both', axis='both', linestyle='--')

            axs[3].legend()
            axs[3].grid(True, which='both', axis='both', linestyle='--')
    

            #     axs[plot_index].set_title(str(slant) + "° relative to north")

            #     data_plot = axs[plot_index].plot(normalized_distances, brightness_values, label = "data")
            #     # smooth_plot = axs[plot_index].plot(normalized_distances, gaussian_filter(brightness_values, sigma= 3), label = "data with gaussian")

            #     axs[plot_index].set_xlabel("Normalized Distance")
            #     axs[plot_index].set_ylabel("Brightness Value")
            #     axs[plot_index].set_xlim(0,1)
            #     axs[plot_index].set_xticks(np.arange(0,1.1,0.1), minor=True)

            #     # y_box.extend(brightness_values)
            #     if index == 0:
            #         handles.extend(data_plot)

            #         # handles.extend(smooth_plot)
            #     if slant_data["meta"]["processing"]["fitted"] == True:
            #         fitted_values = fit_obj.quadratic_limb_darkening(normalized_distances,*list(slant_data["fit"]["quadratic"]["optimal_fit"]["fit_params"].values()))
            #         fit_plot = axs[plot_index].plot(normalized_distances, fitted_values, label = "fit")
            #         if len(handles) < 2:
            #             handles.extend(fit_plot)

            #     # plt.plot(distances, self.limb_darkening_function(
            #         # distances, popt[0], popt[1]))
            #     # Smooth data using moving average

            # #set y_lim
            # # min_box = np.min(y_box); max_box = np.max(y_box); box_range = max_box - min_box
            # # min_box -= box_range * 0.1; max_box += box_range * 0.1

            # for index in range(len(wave_data.keys())):
            #     plot_index = 2+index
            #     axs[plot_index].set_ylim(0.02,0.08)

            fig.tight_layout()

            # if len(handles) == 2:
            #     fig.legend(handles, ["data", "fit"], loc='upper left')  # Set the legend to the top-left corner
            # else:
            #     fig.legend(handles, ["data"], loc='upper left')  # Set the legend to the top-left corner
            # List to keep track of the futures
            fig.savefig(join_strings(self.save_dir, cube_name, band_wave + ".png"), dpi=150)
            # plt.show()
            plt.close()
            time_spent = np.around(time.time() - self.cube_start_time, 3)
            percentage_completed = (bands_done) / leng
            total_time_left = time_spent / percentage_completed - time_spent

            print("Finished quadrants for ", wave_band, "| Spent", time_spent, "so far | expected time left:",
                  np.around(total_time_left, 2), "| total time for cube :",  np.around(total_time_left + time_spent, 3), end="\r")

        time_spent = np.around(time.time() - self.cube_start_time, 3)
        percentage_completed = (cube_index + 1) / self.cube_count
        total_time_left = time_spent / percentage_completed - time_spent
        print("Cube", cube_index + 1, "of", self.cube_count, "| Total time for cube:", time_spent, "seconds | Total Expected time left:",
              np.around(total_time_left, 2), "seconds", "| Total time spent:", np.around(time.time() - self.start_time, 3), "seconds\n")

    def quad_all(self, multi_process: bool = False):
        data = self.get_data()
        self.force_write = (SETTINGS["processing"]["clear_cache"]
                            or SETTINGS["processing"]["redo_figures"])
        self.cube_count = len(data)
        self.start_time = time.time()
        args = []
        for index, (cube_name, cube_data) in enumerate(data.items()):
            self.cube_start_time = time.time()
                        # only important line in this function
            if multi_process:
                args.append([cube_data, index, cube_name])
            else:
                self.gen_cube_quads(cube_data, index, cube_name)
        if multi_process:
            with multiprocessing.Pool(processes=5) as pool:
                pool.starmap(self.gen_cube_quads, args)
