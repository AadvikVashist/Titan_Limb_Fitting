from data_processing.polar_profile import analyze_complete_dataset
from data_processing.sort_and_filter import sort_and_filter
from get_settings import join_strings, check_if_exists_or_write, SETTINGS, get_cumulative_filename
import re
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvims
from data_processing.fitting import fit_data
from scipy.ndimage import gaussian_filter
import multiprocessing
import datetime

def save_fig(fig, save_path):
    fig.savefig(save_path, dpi=150)


class gen_u1_u2_figures:
    def __init__(self, devEnvironment: bool = True):
        if devEnvironment == True:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["dev_figures_sub_path"])
        else:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["prod_figures_sub_path"])
        self.devEnvironment = devEnvironment
        # self.save_dir = join_strings(
        #     SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["plot_sub_path"])
        # self.cube_path = join_strings(
        #     SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"])
        self.selective_fitted_path = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["selected_sub_path"])
        self.fitted_path = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["fitted_sub_path"])
        self.selected_data = self.get_data("selected")
        self.fitted_data = self.get_data("fitted")
        self.cache = SETTINGS["processing"]["clear_cache"]
    def get_fig_path(self, base_path: str, figure_type: str, fig_name: str, cube_name: str):
        figure_type = figure_type
        fig_name = fig_name
        cube_name = cube_name
        now = datetime.datetime.now()
        year = now.year
        month = now.month
        day = now.day
        hour = now.hour
        minute = now.minute
        second = now.second
        try:
            index = 1 + len(os.listdir(base_path))
        except:
            index = 1
        if self.devEnvironment == True:
            file_format = SETTINGS["paths"]["figure_path_format_dev"]
        else:
            file_format = SETTINGS["paths"]["figure_path_format_dev"]

        # Extract variable names from the file_format string
        placeholders = re.findall(r"{(.*?)}", file_format)
        # Create a dictionary to hold variable names and their values
        a = locals()
        file_formatted = '_'.join([str(a[placeholder]) for placeholder in placeholders])
        return join_strings(base_path, file_formatted)
    def emission_to_normalized(self, emission_angle):
        return np.cos(np.deg2rad(emission_angle))

    def get_data(self, type_of_data: str = "selected"):
        """
        takes string of selected or fitted data, then returns all the data
        """
        all_data = {}
        if type_of_data.lower() == "selected":
            base_path = self.selective_fitted_path
            cum_sub_path = get_cumulative_filename("selected_sub_path")
        elif type_of_data.lower() == "fitted":
            base_path = self.fitted_path
            cum_sub_path = get_cumulative_filename("fitted_sub_path")
        else:
            raise ValueError(
                "type_of_data must be selected or fitted, not " + type_of_data)

        if os.path.exists(join_strings(base_path, cum_sub_path)):
            all_data = check_if_exists_or_write(join_strings(
                base_path, cum_sub_path), save=False, verbose=True)
        else:
            cubs = os.listdir(base_path)
            cubs.sort()
            cubs = [cub for cub in cubs if re.fullmatch(
                r'C.*_.*\.pkl', cub) is not None]
            for cub in cubs:
                cube_name = os.path.splitext(cub)[0]
                all_data[cube_name] = check_if_exists_or_write(
                    join_strings(base_path, cub), save=False, verbose=True)
        all_data = dict(sorted(all_data.items()))
        return all_data

    # def gen_cube_quads(self, data: dict, cube_index: int, cube_name: str = None):
    #     """
    #     C*****_1/
    #         0.5µm_1/
    #             0
    #             30
    #             45
    #             ...

    #     """
    #     leng = len(data.keys()) - \
    #         len(SETTINGS["figure_generation"]["unusable_bands"])
    #     if not os.path.exists(join_strings(self.save_dir, cube_name)):
    #         os.makedirs(join_strings(self.save_dir, cube_name))
    #     # cube_vis = pyvims.VIMS(cube_name + "_vis.cub", join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube_name), channel="vis")
    #     # cube_ir = pyvims.VIMS(cube_name + "_ir.cub",  join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube_name), channel="ir")

    #     cube_vis = [data["meta"]["cube_vis"]["bands"][index]
    #                 for index in range(0, 96)]
    #     cube_ir = [data["meta"]["cube_ir"]["bands"][index]
    #                for index in range(0, 352-96)]
    #     bands_done = 1
    #     mpl.rcParams['path.simplify_threshold'] = 1.0
    #     mpl.style.use('fast')
    #     fit_obj = fit_data()
    #     shift = 0
    #     for plot_index, (wave_band, wave_data) in enumerate(data.items()):
    #         if "µm_" not in wave_band:
    #             shift +=1
    #             continue
    #         if plot_index in SETTINGS["figure_generation"]["unusable_bands"]:
    #             continue
    #         else:
    #             bands_done += 1
    #         band_wave = wave_band.split("_")[1] + "_" + wave_band.split("_")[0]
    #         if os.path.exists(join_strings(self.save_dir, cube_name, band_wave + ".png")) and not self.force_write:
    #             continue

    #         fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    #         axs = axs.flatten()

    #         #get the slant data
    #         north_slant = wave_data["north_side"]
    #         south_slant = wave_data["south_side"]
    #         #plot the cube and the lat
    #         if plot_index-shift < 96:
    #             suffix = "_vis"
    #             axs[0].imshow(cube_vis[plot_index-shift], cmap="gray")
    #             pic = data["meta"]["cube_vis"]["lat"]
    #             north_slant_b = np.array([cube_vis[plot_index-shift][pixel_index[0], pixel_index[1]] for pixel_index in north_slant["pixel_indices"]])
    #             south_slant_b = np.array([cube_vis[plot_index-shift][pixel_index[0], pixel_index[1]] for pixel_index in south_slant["pixel_indices"]])
    #         else:
    #             suffix = "_ir"
    #             axs[0].imshow(cube_ir[plot_index-96-shift], cmap="gray")
    #             pic =  data["meta"]["cube_ir"]["lat"]
    #             north_slant_b = np.array([cube_ir[plot_index-shift-96][pixel_index[0], pixel_index[1]] for pixel_index in north_slant["pixel_indices"]])
    #             south_slant_b = np.array([cube_ir[plot_index-shift-96][pixel_index[0], pixel_index[1]] for pixel_index in south_slant["pixel_indices"]])
    #         if not np.all(north_slant_b == north_slant["brightness_values"]):
    #             raise ValueError("Brightness values do not match")
    #         if not np.all(south_slant_b == south_slant["brightness_values"]):
    #             raise ValueError("Brightness values do not match")
    #         for pixel_index in north_slant["pixel_indices"]:
    #             pic[pixel_index[0], pixel_index[1]] = -90
    #         for pixel_index in south_slant["pixel_indices"]:
    #             pic[pixel_index[0], pixel_index[1]] = -90
    #         axs[1].imshow(pic, cmap="gray")
    #         shape = data["meta"]["cube_ir"]["lat"].shape
    #         axs[1].plot([data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][1] + np.sin(np.radians(data["meta"]["north_orientation" + suffix])) * 50],
    #                     [data["meta"]["center_of_cube" + suffix][0], data["meta"]["center_of_cube" + suffix][0] - np.cos(np.radians(data["meta"]["north_orientation" + suffix])) * 50], color=(0,0,0), label="North")

    #         axs[1].plot([data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][1] - np.sin(np.radians(data["meta"]["north_orientation" + suffix])) * 50],
    #                     [data["meta"]["center_of_cube" + suffix][0], data["meta"]["center_of_cube" + suffix][0] + np.cos(np.radians(data["meta"]["north_orientation" + suffix])) * 50], color=(0.5,0.5,0.5), label="South")

    #         axs[1].scatter(data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][0], color=(1,0,0), s = 60 , label="Center of Disk")
    #         axs[1].scatter(data["meta"]["lowest_inc_location" + suffix][1], data["meta"]["lowest_inc_location" + suffix][0], color=(0.2,0.5,0.7), s = 60 , label="Lowest Inc")

    #         axs[0].set_title(cube_name + " " + wave_band)
    #         axs[1].set_title(cube_name + " lat")
    #         axs[1].set_xlim(0, shape[1]-1)
    #         axs[1].set_ylim(shape[0]-1, 0)

    #         #put the slant data on to the lat array
    #         area = len([dat for dat in data["meta"]["cube_vis"]["ground"].flatten() if dat != False])
    #         length = np.sqrt(area / np.pi)

    #         axs[1].plot([data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][1] + np.sin(np.radians(data["meta"]["north_orientation" + suffix] + north_slant["angle"])) * length],
    #                     [data["meta"]["center_of_cube" + suffix][0], data["meta"]["center_of_cube" + suffix][0] - np.cos(np.radians(data["meta"]["north_orientation" + suffix] + north_slant["angle"])) * length], color=(0.2,0.6,0.3), label="north_side_profile")
    #         axs[1].plot([data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][1] + np.sin(np.radians(data["meta"]["north_orientation" + suffix] + south_slant["angle"])) * length],
    #                     [data["meta"]["center_of_cube" + suffix][0], data["meta"]["center_of_cube" + suffix][0] - np.cos(np.radians(data["meta"]["north_orientation" + suffix] + south_slant["angle"])) * length], color=(0.2,0.5,0.6), label="south_side_profile")
    #         axs[1].legend(loc="upper left")

    #         #start bottom row of plots
    #         axs[2].set_title(cube_name + "  " + wave_band + " " + str(north_slant["angle"]) + "° relative to north")
    #         axs[3].set_title(cube_name + "  " + wave_band + " " + str(south_slant["angle"]) + "° relative to north")

    #         #figure out what the best y range is for the plot
    #         range_vals = list(north_slant["brightness_values"]); range_vals.extend(south_slant["brightness_values"])
    #         y_range = np.max(range_vals) - np.min(range_vals)
    #         y_min = np.min(range_vals) - y_range * 0.1; y_min -= y_min % 0.01
    #         y_max = np.max(range_vals) + y_range * 0.1; y_max += y_max % 0.01

    #         if y_range < 0.05:
    #             axs[2].set_yticks(np.arange(y_min,y_max+0.001,0.0025), minor=True)
    #             axs[3].set_yticks(np.arange(y_min,y_max+0.001,0.0025), minor=True)

    #         #set plot styles

    #         axs[2].set_xlabel("Normalized Distance")
    #         axs[2].set_ylabel("Brightness Value")
    #         axs[2].set_ylim(y_min,y_max)
    #         axs[2].set_xlim(0,1)
    #         axs[2].set_yticks(np.arange(y_min,y_max+0.001,0.01))
    #         axs[2].set_xticks(np.arange(0,1.1,0.1))
    #         axs[2].set_xticks(np.arange(0,1.1,0.05), minor=True)

    #         axs[3].set_xlabel("Normalized Distance")
    #         axs[3].set_ylabel("Brightness Value")
    #         axs[3].set_ylim(y_min,y_max)

    #         axs[3].set_xlim(0,1)
    #         axs[3].set_yticks(np.arange(y_min,y_max+0.001,0.01))
    #         axs[3].set_xticks(np.arange(0,1.1,0.1))
    #         axs[3].set_xticks(np.arange(0,1.1,0.05), minor=True)

    #         #plot the data
    #         normalized_distances = self.emission_to_normalized(emission_angle=north_slant["emission_angles"])
    #         axs[2].plot(normalized_distances, north_slant["brightness_values"], color= (0.2,0.6,0.3), label="north_side_profile")

    #         normalized_distances = self.emission_to_normalized(emission_angle=south_slant["emission_angles"])
    #         axs[3].plot(normalized_distances, south_slant["brightness_values"], color=(0.2,0.5,0.6), label="south_side_profile")

    #         #work with fits
    #         if north_slant["meta"]["processing"]["fitted"] == True and len(north_slant["fit"]["quadratic"]["optimal_fit"]) != 0:
    #             normalized_distances = self.emission_to_normalized(emission_angle=north_slant["emission_angles"])
    #             fitted_values = fit_obj.quadratic_limb_darkening(normalized_distances,*list(north_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].values()))
    #             axs[2].plot(normalized_distances, fitted_values, color= (0.9,0.2,0.2), label = "fit")
    #             string = '\n'.join([str(key) + " : " + str(item) for key,item in north_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].items()])  + "\n r2_score: " + str(north_slant["fit"]["quadratic"]["optimal_fit"]["r2"])
    #             axs[2].text(0.95, 0.03, string,
    #                     verticalalignment='bottom', horizontalalignment='right',
    #                     transform=axs[2].transAxes,
    #                     color='black', fontsize=10,
    #                     bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})

    #         if south_slant["meta"]["processing"]["fitted"] == True  and len(south_slant["fit"]["quadratic"]["optimal_fit"]) != 0:
    #             normalized_distances = self.emission_to_normalized(emission_angle=south_slant["emission_angles"])
    #             fitted_values = fit_obj.quadratic_limb_darkening(normalized_distances,*list(south_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].values()))
    #             axs[3].plot(normalized_distances, fitted_values,color= (0.9,0.2,0.2), label = "fit")
    #             string = '\n'.join([str(key) + " : " + str(item) for key,item in south_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].items()])  + "\n r2_score: " + str(south_slant["fit"]["quadratic"]["optimal_fit"]["r2"])
    #             axs[3].text(0.95, 0.03, string,
    #                     verticalalignment='bottom', horizontalalignment='right',
    #                     transform=axs[3].transAxes,
    #                     color='black', fontsize=10,
    #                     bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 5})

    #         axs[2].legend()
    #         axs[2].grid(True, which='both', axis='both', linestyle='--')

    #         axs[3].legend()
    #         axs[3].grid(True, which='both', axis='both', linestyle='--')

    #         #     axs[plot_index].set_title(str(slant) + "° relative to north")

    #         #     data_plot = axs[plot_index].plot(normalized_distances, brightness_values, label = "data")
    #         #     # smooth_plot = axs[plot_index].plot(normalized_distances, gaussian_filter(brightness_values, sigma= 3), label = "data with gaussian")

    #         #     axs[plot_index].set_xlabel("Normalized Distance")
    #         #     axs[plot_index].set_ylabel("Brightness Value")
    #         #     axs[plot_index].set_xlim(0,1)
    #         #     axs[plot_index].set_xticks(np.arange(0,1.1,0.1), minor=True)

    #         #     # y_box.extend(brightness_values)
    #         #     if index == 0:
    #         #         handles.extend(data_plot)

    #         #         # handles.extend(smooth_plot)
    #         #     if slant_data["meta"]["processing"]["fitted"] == True:
    #         #         fitted_values = fit_obj.quadratic_limb_darkening(normalized_distances,*list(slant_data["fit"]["quadratic"]["optimal_fit"]["fit_params"].values()))
    #         #         fit_plot = axs[plot_index].plot(normalized_distances, fitted_values, label = "fit")
    #         #         if len(handles) < 2:
    #         #             handles.extend(fit_plot)

    #         #     # plt.plot(distances, self.limb_darkening_function(
    #         #         # distances, popt[0], popt[1]))
    #         #     # Smooth data using moving average

    #         # #set y_lim
    #         # # min_box = np.min(y_box); max_box = np.max(y_box); box_range = max_box - min_box
    #         # # min_box -= box_range * 0.1; max_box += box_range * 0.1

    #         # for index in range(len(wave_data.keys())):
    #         #     plot_index = 2+index
    #         #     axs[plot_index].set_ylim(0.02,0.08)

    #         fig.tight_layout()

    #         # if len(handles) == 2:
    #         #     fig.legend(handles, ["data", "fit"], loc='upper left')  # Set the legend to the top-left corner
    #         # else:
    #         #     fig.legend(handles, ["data"], loc='upper left')  # Set the legend to the top-left corner
    #         # List to keep track of the futures
    #         fig.savefig(join_strings(self.save_dir, cube_name, band_wave + ".png"), dpi=150)
    #         # plt.show()
    #         plt.close()
    #         time_spent = np.around(time.time() - self.cube_start_time, 3)
    #         percentage_completed = (bands_done) / leng
    #         total_time_left = time_spent / percentage_completed - time_spent

    #         print("Finished quadrants for ", wave_band, "| Spent", time_spent, "so far | expected time left:",
    #               np.around(total_time_left, 2), "| total time for cube :",  np.around(total_time_left + time_spent, 3), end="\r")

    #     time_spent = np.around(time.time() - self.cube_start_time, 3)
    #     percentage_completed = (cube_index + 1) / self.cube_count
    #     total_time_left = time_spent / percentage_completed - time_spent
    #     print("Cube", cube_index + 1, "of", self.cube_count, "| Total time for cube:", time_spent, "seconds | Total Expected time left:",
    #           np.around(total_time_left, 2), "seconds", "| Total time spent:", np.around(time.time() - self.start_time, 3), "seconds\n")

    def u_vs_wave(self, cube_data, cube_name, cube_index, force_write):
        """
        C*****_1/
            0.5µm_1/
                0
                30
                45
                ...

        """
        leng = len(cube_data.keys()) - \
            len(SETTINGS["figure_generation"]["unusable_bands"])
        if not os.path.exists(join_strings(self.save_dir, cube_name)):
            os.makedirs(join_strings(self.save_dir, cube_name))
        # cube_vis = pyvims.VIMS(cube_name + "_vis.cub", join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube_name), channel="vis")
        # cube_ir = pyvims.VIMS(cube_name + "_ir.cub",  join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube_name), channel="ir")
        cube_vis = [cube_data["meta"]["cube_vis"]["bands"][index]
                    for index in range(0, 96)]
        cube_ir = [cube_data["meta"]["cube_ir"]["bands"][index]
                   for index in range(0, 352-96)]

        fit_obj = fit_data()
        base_path = join_strings(self.save_dir,SETTINGS["paths"]["figure_subpath"]["u_vs_wavelength"] )
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        fig = plt.figure(figsize=(12, 7))
        plt.title("Plots of Quadratic LDL coeffs from " + cube_name)
        shift = 0

        ind = 0
        data = {}
        plt.rcParams['font.family'] = 'serif'
        for plot_index, (wave_band, wave_data) in enumerate(cube_data.items()):
            if "µm_" not in wave_band:
                shift += 1
                continue
            if plot_index in SETTINGS["figure_generation"]["unusable_bands"]:
                ind += 1
                continue
            band_wave = wave_band.split("_")[1] + "_" + wave_band.split("_")[0]

            if ind not in data:
                data[ind] = {"wave": [], 
                            "north_u1": [], "north_u2": [], "south_u1": [], "south_u2": [],
                            "north u1 + u2": [], "north u1 + 2*u2": [], "north u1 - u2": [],
                            "south u1 + u2": [], "south u1 + 2*u2": [], "south u1 - u2": [],
                            "north - south u1 + u2": []}
            data[ind]["wave"].append(float(wave_band.split("_")[0].replace("µm", "")))
            # get the slant data
            north_slant = wave_data["north_side"]
            south_slant = wave_data["south_side"]

            # plot the cube and the lat
            if plot_index-shift < 96:
                pic = cube_data["meta"]["cube_vis"]["lat"]
                north_slant_b = np.array([cube_vis[plot_index-shift][pixel_index[0], pixel_index[1]]
                                         for pixel_index in north_slant["pixel_indices"]])
                south_slant_b = np.array([cube_vis[plot_index-shift][pixel_index[0], pixel_index[1]]
                                         for pixel_index in south_slant["pixel_indices"]])
            else:
                pic = cube_data["meta"]["cube_ir"]["lat"]
                north_slant_b = np.array([cube_ir[plot_index-shift-96][pixel_index[0], pixel_index[1]]
                                         for pixel_index in north_slant["pixel_indices"]])
                south_slant_b = np.array([cube_ir[plot_index-shift-96][pixel_index[0], pixel_index[1]]
                                         for pixel_index in south_slant["pixel_indices"]])
            if not np.all(north_slant_b == north_slant["brightness_values"]):
                raise ValueError("Brightness values do not match")
            if not np.all(south_slant_b == south_slant["brightness_values"]):
                raise ValueError("Brightness values do not match")

            try:
                nu1 = north_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u1"]
                nu2 = north_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u2"]
            except:
                nu1 = np.nan
                nu2 = np.nan
            try:
                su1 = south_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u1"]
                su2 = south_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u2"]
            except:
                su1 = np.nan
                su2 = np.nan

            data[ind]["north_u1"].append(nu1)
            data[ind]["north_u2"].append(nu2)
            data[ind]["south_u1"].append(su1)
            data[ind]["south_u2"].append(su2)

            data[ind]["north u1 + u2"].append(nu1+nu2)
            data[ind]["north u1 - u2"].append(nu1-nu2)
            data[ind]["north u1 + 2*u2"].append(nu1+2*nu2)
            data[ind]["south u1 + u2"].append(su1+su2)
            data[ind]["south u1 - u2"].append(su1-su2)
            data[ind]["south u1 + 2*u2"].append(su1+2*su2)
            data[ind]["north - south u1 + u2"].append(nu1+nu2 - (su1+su2))
            
        # fig.savefig(join_strings(self.save_dir, cube_name, band_wave + ".png"), dpi=150)
        # plt.show()
        # plt.scatter(wvlng, north_u1, label = "north u1")
        # plt.scatter(wvlng, north_u2, label = "north u2")
        # plt.scatter(wvlng, south_u1, label = "south u1")
        # plt.scatter(wvlng, south_u2, label = "south u2")

        # sorted_wvlng, sorted_n_plus, sorted_s_plus = zip(
        #     *sorted(zip(wvlng, north_u1_plus_u2, south_u1_plus_u2)))
        # # plt.plot(sorted_wvlng, gaussian_filter(sorted_n_plus, sigma = 5), label = "north u1 + u2")
        # # plt.plot(sorted_wvlng, gaussian_filter(sorted_n_minus, sigma = 5), label = "north u1 + 2*u2")
        # # plt.plot(sorted_wvlng,  gaussian_filter(sorted_s_plus, sigma = 5), label = "south u1 + u2")
        # # plt.plot(sorted_wvlng, sorted_s_minus, label = "south u1 + 2*u2")
        # # plt.plot(sorted_wvlng, sorted_n_two_plus, label = "north u1 + 2*u2")
        # # plt.plot(sorted_wvlng, gaussian_filter(sorted_n_plus, sigma = 5), label = "north u1 + u2")
        # plt.plot(sorted_wvlng, sorted_n_plus,
        #          label="North | µ1 + µ2", color=(1, 0.78, 0.33))
        # plt.plot(sorted_wvlng, sorted_s_plus,
        #          label="South | µ1 + µ2", color=(0.31, 0.72, 0.73))
        # plt.plot(sorted_wvlng, np.array(sorted_n_plus) -
        #          np.array(sorted_s_plus), label="North-South | µ1 + µ2", color="red")
        for index, (key, val) in enumerate(data.items()):
            if index == 0:
                plt.plot(val["wave"], val["north u1 + u2"],
                        label="North | µ1 + µ2", color=(1, 0.78, 0.33))
                plt.plot(val["wave"], val["south u1 + u2"],
                        label="South | µ1 + µ2", color=(0.31, 0.72, 0.73))
                plt.plot(val["wave"], val["north - south u1 + u2"] , label="North-South | µ1 + µ2", color="red")
            else:
                plt.plot(val["wave"], val["north u1 + u2"],
                        color=(1, 0.78, 0.33))
                plt.plot(val["wave"], val["south u1 + u2"],
                        color=(0.31, 0.72, 0.73))
                plt.plot(val["wave"], val["north - south u1 + u2"], color="red")
        plt.xticks(np.arange(0.25, 3.3, 0.25),  labels=[str(x) + "µm" for x in np.arange(0.25, 3.3, 0.25)])
        plt.xticks(np.arange(0.25, 3.3, 0.0625), minor=True)
        #find the min and max of the y axis
        ymin = np.min([np.min((val["north u1 + u2"],val["south u1 + u2"],val["north - south u1 + u2"]) ) for key, val in data.items()])
        ymax = np.max([np.max((val["north u1 + u2"],val["south u1 + u2"],val["north - south u1 + u2"]) ) for key, val in data.items()])
        #round ymin down to the nearest 0.25
        ymin -= ymin % 0.25
        ymax -= ymax % 0.25 - 0.25
        plt.xlim(0.25, 3.25)
        plt.ylim(ymin, ymax)
        plt.yticks(np.arange(ymin, ymax + 0.001, 0.25))
        plt.yticks(np.arange(ymin, ymax + 0.001, 0.0625), minor=True)
        # plt.plot(sorted_wvlng, sorted_s_two_plus, label = "south u1 + 2*u2")
        plt.axhline(y=0, color='black', linestyle='--')
        plt.text(0.5, 0.98, "Limb Darkening", transform=plt.gca().transAxes,
                ha='center', va='top', fontsize=18)
        plt.text(0.5, 0.05, "Limb Brightening", transform=plt.gca().transAxes,
                ha='center', va='top', fontsize=18)

        
        plt.xlabel("Wavelength (µm)")
        plt.ylabel("Parameter Value")
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # plt.show()
        path = self.get_fig_path(base_path, "u_vs_wave", "DPS_55", cube_name=cube_name) + ".png"
        print("Saving figure to", path)
        fig.savefig(path, dpi=150)
        plt.close()
        #     time_spent = np.around(time.time() - self.cube_start_time, 3)
        #     percentage_completed = (bands_done) / leng
        #     total_time_left = time_spent / percentage_completed - time_spent

        #     print("Finished quadrants for ", wave_band, "| Spent", time_spent, "so far | expected time left:",
        #           np.around(total_time_left, 2), "| total time for cube :",  np.around(total_time_left + time_spent, 3), end="\r")

        # time_spent = np.around(time.time() - self.cube_start_time, 3)
        # percentage_completed = (cube_index + 1) / self.cube_count
        # total_time_left = time_spent / percentage_completed - time_spent
        # print("Cube", cube_index + 1, "of", self.cube_count, "| Total time for cube:", time_spent, "seconds | Total Expected time left:",
        #       np.around(total_time_left, 2), "seconds", "| Total time spent:", np.around(time.time() - self.start_time, 3), "seconds\n")

    def gen_u_vs_wavelength(self, multi_process: bool = False, data=None):
        # SETTINGS["processing"]["redo_figures"])
        # if data == None:
        data = self.selected_data
        args = []
        force_write = self.cache or SETTINGS["processing"]["redo_u_vs_wavelength_figure_generation"]
        for index, (cube_name, cube_data) in enumerate(data.items()):
            self.cube_start_time = time.time()
            # only important line in this function
            if multi_process:
                args.append([cube_data, cube_name, index, force_write])
            else:
                self.u_vs_wave(cube_data, cube_name,  index, force_write)
        if multi_process:
            with multiprocessing.Pool(processes=5) as pool:
                pool.starmap(self.u_vs_wave, args)

    # def gen_u_vs_wavelength(self, multi_process: bool = False, data = None):
    #     # SETTINGS["processing"]["redo_figures"])
    #     if data == None:
    #         data == self.selected_data
    #     args = []
    #     force_write = self.cache or SETTINGS["processing"]["redo_u_vs_wavelength_figure_generation"]
    #     for index, (cube_name, cube_data) in enumerate(data.items()):
    #         self.cube_start_time = time.time()
    #                     # only important line in this function
    #         if multi_process:
    #             args.append([cube_data, index, cube_name,force_write])
    #         else:
    #             self.u_vs_wave(cube_data, index, cube_name, force_write)
    #     if multi_process:
    #         with multiprocessing.Pool(processes=5) as pool:
    #             pool.starmap(self.u_vs_wave, args)
    def u1_u2_all_figures(self, multi_process: bool = False):

        # or SETTINGS["processing"]["redo_figures"])
        # self.cube_count = len(data)
        self.gen_u_vs_wavelength(multi_process=multi_process)
        self.start_time = time.time()
        # args = []

        # for index, (cube_name, cube_data) in enumerate(data.items()):
        #     self.cube_start_time = time.time()
        #                 # only important line in this function
        #     if multi_process:
        #         args.append([cube_data, index, cube_name])
        #     else:
        #         self.gen_cube_quads(cube_data, index, cube_name)
        # if multi_process:
        #     with multiprocessing.Pool(processes=5) as pool:
        #         pool.starmap(self.gen_cube_quads, args)
