
from settings.get_settings import join_strings, check_if_exists_or_write, SETTINGS, get_cumulative_filename
import re
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvims
from ..data_processing.fitting import fit_data
from scipy.ndimage import gaussian_filter
import multiprocessing
from typing import Union
import datetime
from PIL import Image

def rotate_theta(band, theta):
    """
    Rotates the theta values by the given theta
    """
    matrix = np.zeros(band.shape*11)
    center_of_matrix = np.array(matrix.shape -1)/2
    plt.imshow(matrix)
    plt.show()
    return (band + theta) % 360       
def save_fig(fig, save_path):
    fig.savefig(save_path, dpi=150)


class gen_u1_u2_figures:
    def __init__(self, devEnvironment: bool = True):
        if devEnvironment == True:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["dev_figures_sub_path"])
        else:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["prod_figures_sub_path"])
        self.devEnvironment = devEnvironment
        self.selective_fitted_path = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["selected_sub_path"])
        self.fitted_path = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["fitted_sub_path"])
        self.selected_data = self.get_data("selected")
        self.fitted_data = self.get_data("fitted")
        self.cache = SETTINGS["processing"]["clear_cache"]

    def get_fig_path(self, base_path: str, figure_type: str, fig_name: str, cube_name: str, ind_recog: bool = True):
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
        # check how many of the figures already exist
        if ind_recog:
            try:
                vals = [path for path in os.listdir(
                    base_path) if cube_name in path and fig_name in path]
                index = 1 + len(vals)
            except:
                index = 1
        else:
            index = 1
        if self.devEnvironment == True:
            file_format = SETTINGS["paths"]["figure_path_format_dev"]
        else:
            file_format = SETTINGS["paths"]["figure_path_format"]

        # Extract variable names from the file_format string
        placeholders = re.findall(r"{(.*?)}", file_format)
        # Create a dictionary to hold variable names and their values
        a = locals()
        file_formatted = '_'.join([str(a[placeholder])
                                  for placeholder in placeholders])
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
        cube_vis = [cube_data["meta"]["cube_vis"]["bands"][index]
                    for index in range(0, 96)]
        cube_ir = [cube_data["meta"]["cube_ir"]["bands"][index]
                   for index in range(0, 352-96)]

        fit_obj = fit_data()
        base_path = join_strings(
            self.save_dir, SETTINGS["paths"]["figure_subpath"]["u_vs_wavelength"])
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        fig = plt.figure(figsize=(12, 7))
        shift = 0

        ind = 0
        data = {}
        converse = {}
        # plt.rcParams['font.family'] = 'serif'
        for plot_index, (wave_band, wave_data) in enumerate(cube_data.items()):
            if "µm_" not in wave_band:
                shift += 1
                continue
            if plot_index in SETTINGS["figure_generation"]["unusable_bands"] or (plot_index - shift + 1 > 96 and plot_index - shift + 1< 108):
                ind += 1
                continue
            band_wave = wave_band.split("_")[1] + "_" + wave_band.split("_")[0]

            if ind not in data:
                data[ind] = {"wave": [],
                             "north_u1": [], "north_u2": [], "south_u1": [], "south_u2": [],
                             "north u1 + u2": [], "north u1 + 2*u2": [], "north u1 - u2": [],
                             "south u1 + u2": [], "south u1 + 2*u2": [], "south u1 - u2": [],
                             "north - south u1 + u2": []}
            data[ind]["wave"].append(
                float(wave_band.split("_")[0].replace("µm", "")))
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
        lists =  list(data.keys())

        for index, (key, val) in enumerate(data.items()):
            if index == 0:
                plt.plot(val["wave"], val["north u1 + u2"], label="North | µ1 + µ2", linewidth = 2, color=(1, 0.78, 0.33))
                plt.plot(val["wave"], val["south u1 + u2"], label="South | µ1 + µ2", linewidth = 2, color=(0.23, 0.54, 0.54))
                # plt.plot(val["wave"], val["north - south u1 + u2"], label="North-South | µ1 + µ2", linewidth = 2, color="red")
            else:
                plt.plot(val["wave"], val["north u1 + u2"], linewidth = 2, color=(1, 0.78, 0.33))
                plt.plot(val["wave"], val["south u1 + u2"], linewidth = 2, color=(0.23, 0.54, 0.54))
                # plt.plot(val["wave"], val["north - south u1 + u2"], linewidth = 2, color="red")
            if index < len(lists) - 1:
                vals = data[lists[index+1]]
                inbetween = {"wave": (val["wave"][-1], data[list(data.keys())[index+1]]["wave"][0]),
                            "north u1 + u2": (val["north u1 + u2"][-1], data[list(data.keys())[index+1]]["north u1 + u2"][0]),
                            "north u1 + 2*u2": (val["north u1 + 2*u2"][-1], data[list(data.keys())[index+1]]["north u1 + 2*u2"][0]),
                            "north u1 - u2": (val["north u1 - u2"][-1], data[list(data.keys())[index+1]]["north u1 - u2"][0]),
                            "south u1 + u2": (val["south u1 + u2"][-1], data[list(data.keys())[index+1]]["south u1 + u2"][0]),
                            "south u1 + 2*u2": (val["south u1 + 2*u2"][-1], data[list(data.keys())[index+1]]["south u1 + 2*u2"][0]),
                            "south u1 - u2": (val["south u1 - u2"][-1], data[list(data.keys())[index+1]]["south u1 - u2"][0]),
                            "north - south u1 + u2": (val["north - south u1 + u2"][-1], data[list(data.keys())[index+1]]["north - south u1 + u2"][0])
                            }
                plt.plot(inbetween["wave"], inbetween["north u1 + u2"], color=(0.5, 0.39, 0.165), linestyle = "--", linewidth = 1)
                plt.plot(inbetween["wave"], inbetween["south u1 + u2"], color=(0.15, 0.36, 0.365), linestyle = "--", linewidth = 1)
                # plt.plot(inbetween["wave"], inbetween["north - south u1 + u2"],  color=(0.5,0,0), linestyle = "--", linewidth = 1)
        plt.xticks(np.arange(0.25, 3.3, 0.25),  labels=[
                   str(x) + "µm" for x in np.arange(0.25, 3.3, 0.25)])
        plt.xticks(np.arange(0.25, 3.3, 0.0625), minor=True)
        # find the min and max of the y axis
        ymin = np.nanmin([np.nanmin((val["north u1 + u2"], val["south u1 + u2"],
                      val["north - south u1 + u2"])) for key, val in data.items()])
        ymax = np.nanmax([np.nanmax((val["north u1 + u2"], val["south u1 + u2"],
                      val["north - south u1 + u2"])) for key, val in data.items()])
        # round ymin down to the nearest 0.25
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

        plt.xlabel("Wavelength (µm)", fontsize=14)
        plt.ylabel("µ1 + µ2", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # plt.show()
        path = self.get_fig_path(
            base_path, "u_vs_wave", "DPS_55", cube_name=cube_name) + ".png"
        print("Saving figure to", path)
        fig.savefig(path, dpi=150)
        plt.close()

    def gen_u_vs_wavelength(self, multi_process: Union[bool, int] = False, data=None):
        # SETTINGS["processing"]["redo_figures"])
        if data == None:
            data = self.selected_data


        if multi_process == True or multi_process >= 1:
            args =[]
        
        if multi_process == True:
            multi_process_core_count = 3 # default val
        elif type(multi_process) == int:
            multi_process_core_count = multi_process
        if multi_process_core_count == 1:
            multi_process = False


        force_write = self.cache or SETTINGS["processing"]["redo_u_vs_wavelength_figure_generation"]
        for index, (cube_name, cube_data) in enumerate(data.items()):
            self.cube_start_time = time.time()
            # only important line in this function
            if multi_process:
                args.append([cube_data, cube_name, index, force_write])
            else:
                self.u_vs_wave(cube_data, cube_name,  index, force_write)
        if multi_process:
            with multiprocessing.Pool(processes=multi_process_core_count) as pool:
                pool.starmap(self.u_vs_wave, args)
    
    def get_time(self, datetime_var):
        
        # Get the start of the year for the same year as the given datetime
        start_of_year = datetime.datetime(datetime_var.year, 1, 1, tzinfo=datetime.timezone.utc)

        # Calculate the time difference in seconds between the given datetime and the start of the year
        time_difference_seconds = (datetime_var - start_of_year).total_seconds()

        # Calculate the total number of seconds in a year (considering leap years)
        total_seconds_in_year = 366 * 24 * 60 * 60 if datetime_var.year % 4 == 0 else 365 * 24 * 60 * 60

        # Calculate the percentage of the year
        percentage_of_year = (time_difference_seconds / total_seconds_in_year)
        return datetime_var.year + percentage_of_year
    def u_vs_time_averages(self, cubes, band, force_write):
        """
        C*****_1/
            0.5µm_1/
                0
                30
                45
                ...

        """
        
        
        band_ind = [(band_name,index) for index, band_name in enumerate(list(cubes[list(cubes.keys())[0]].keys())) if "µm" in band_name and str(band) in band_name.split("_")[1]]
        band_index = band_ind[0][1]
        band_ind = band_ind[0][0]
        min_band = band_index - 5
        max_band = band_index + 5

        base_path = join_strings(
            self.save_dir, SETTINGS["paths"]["figure_subpath"]["u_vs_time_per_wavelength"])
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        fig = plt.figure(figsize=(12, 3))
        # plt.title("Plots of Quadratic LDL coeffs at " + band_ind)
        shift = 0

        ind = 0
        data = {"time": [],
            "north_u1": [], "north_u2": [], "south_u1": [], "south_u2": [],
            "north u1 + u2": [], "north u1 + 2*u2": [], "north u1 - u2": [],
            "south u1 + u2": [], "south u1 + 2*u2": [], "south u1 - u2": [],
            "north - south u1 + u2": []}        
        converse = {}
        # plt.rcParams['font.family'] = 'serif'
        band_wave = band_ind.split("_")[1] + "_" + band_ind.split("_")[0]

        path = self.get_fig_path(
            base_path, "u_vs_time_average", "DPS_55", cube_name=band_wave, ind_recog = False) + ".png"
        if os.path.exists(path) and not force_write:
            return
        for cube, cube_data in cubes.items():
            cube_vis = [cube_data["meta"]["cube_vis"]["bands"][index] for index in range(0, 96)]
            cube_ir = [cube_data["meta"]["cube_ir"]["bands"][index] for index in range(0, 352-96)]
            band_ind = [(band_name,index) for index, band_name in enumerate(list(cubes[list(cubes.keys())[0]].keys())) if "µm" in band_name and str(band) in band_name.split("_")[1]]
            band_index = band_ind[0][1]
            band_ind = band_ind[0][0]
            min_band = band_index - 9 
            max_band = band_index + 9
            # band_wave = wave_band.split("_")[1] + "_" + wave_band.split("_")[0]
            nu1 = []; nu2 = []; su1 = []; su2 = []
            for bands in range(min_band, max_band):
                
                wave_data = cube_data[list(cube_data.keys())[bands]]
                north_slant = wave_data["north_side"]
                south_slant = wave_data["south_side"]

                try:
                    nu1.append(north_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u1"])
                    nu2.append(north_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u2"])
                except:
                    nu1.append(np.nan)
                    nu2.append(np.nan)
                try:
                    su1.append(south_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u1"])
                    su2.append(south_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u2"])
                except:
                    su1.append(np.nan)
                    su2.append(np.nan)
            nu1 = np.nanmean(nu1)
            nu2 = np.nanmean(nu2)
            su1 = np.nanmean(su1)
            su2 = np.nanmean(su2)
            
            data["time"].append(self.get_time(cube_data["meta"]["cube_vis"]["time"]))
            data["north_u1"].append(nu1)
            data["north_u2"].append(nu2)
            data["south_u1"].append(su1)
            data["south_u2"].append(su2)

            data["north u1 + u2"].append(nu1+nu2)
            data["north u1 - u2"].append(nu1-nu2)
            data["north u1 + 2*u2"].append(nu1+2*nu2)
            data["south u1 + u2"].append(su1+su2)
            data["south u1 - u2"].append(su1-su2)
            data["south u1 + 2*u2"].append(su1+2*su2)
            data["north - south u1 + u2"].append(nu1+nu2 - (su1+su2))
        lists =  list(data.keys())


        plt.plot(data["time"], data["north u1 + u2"], label="North | µ1 + µ2", linewidth = 2, color=(1, 0.78, 0.33), marker ="o")
        plt.plot(data["time"], data["south u1 + u2"], label="South | µ1 + µ2", linewidth = 2, color=(0.23, 0.54, 0.54), marker ="o")
        # plt.plot(data["time"], data["north - south u1 + u2"], label="North-South | µ1 + µ2", linewidth = 2, color="red", marker ="s")
        # plt.plot(data["time"], data["north u1 + u2"], linewidth = 2, color=(1, 0.78, 0.33))
        # plt.plot(data["time"], data["south u1 + u2"], linewidth = 2, color=(0.31, 0.72, 0.73))
        # plt.plot(data["time"], data["north - south u1 + u2"], linewidth = 2, color="red")
        ymin = np.nanmin((data["north u1 + u2"], data["south u1 + u2"],
                      data["north - south u1 + u2"]))
        ymax = np.nanmax((data["north u1 + u2"], data["south u1 + u2"],
                      data["north - south u1 + u2"]))
        # round ymin down to the nearest 0.25
        ymin -= ymin % 0.25 + 0.25
        ymax -= ymax % 0.25 - 0.25
        # plt.xlim(0.25, 3.25)
        plt.ylim(ymin, ymax)
        if ymax - ymin > 3:
            max_tick = 0.5
        else:
            max_tick = 0.25
        plt.yticks(np.arange(ymin, ymax + 0.001, max_tick))
        plt.yticks(np.arange(ymin, ymax + 0.001, max_tick/4), minor=True)
        # plt.plot(sorted_wvlng, sorted_s_two_plus, label = "south u1 + 2*u2")
        plt.axhline(y=0, color='black', linestyle='--')
        plt.text(0.4, 0.98, "Limb Darkening", transform=plt.gca().transAxes,
                 ha='center', va='top', fontsize=14)
        plt.text(0.4, 0.12, "Limb Brightening", transform=plt.gca().transAxes,
                 ha='center', va='top', fontsize=14)

        plt.xlabel("Time", fontsize=14)
        plt.ylabel("µ1 + µ2", fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()

        # plt.show()

        print("Saving figure to", path)
        plt.show()
        fig.savefig(path, dpi=150)
        # plt.show()
        plt.close()
        
    def u_vs_time(self, cubes, band, force_write):
        """
        C*****_1/
            0.5µm_1/
                0
                30
                45
                ...

        """
        band_ind = [band_name for index, band_name in enumerate(list(cubes[list(cubes.keys())[0]].keys())) if "µm" in band_name and str(band) in band_name.split("_")[1]]
        band_ind = band_ind[0]


        base_path = join_strings(
            self.save_dir, SETTINGS["paths"]["figure_subpath"]["u_vs_time_per_wavelength"])
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        fig = plt.figure(figsize=(12, 7))
        plt.title("Plots of Quadratic LDL coeffs at " + band_ind)
        shift = 0

        ind = 0
        data = {"time": [],
            "north_u1": [], "north_u2": [], "south_u1": [], "south_u2": [],
            "north u1 + u2": [], "north u1 + 2*u2": [], "north u1 - u2": [],
            "south u1 + u2": [], "south u1 + 2*u2": [], "south u1 - u2": [],
            "north - south u1 + u2": []}        
        converse = {}
        plt.rcParams['font.family'] = 'serif'
        band_wave = band_ind.split("_")[1] + "_" + band_ind.split("_")[0]

        path = self.get_fig_path(
            base_path, "u_vs_time", "DPS_55", cube_name=band_wave, ind_recog = False) + ".png"
        if os.path.exists(path) and not force_write:
            return
        for cube, cube_data in cubes.items():
            cube_vis = [cube_data["meta"]["cube_vis"]["bands"][index]
                    for index in range(0, 96)]
            cube_ir = [cube_data["meta"]["cube_ir"]["bands"][index]
                   for index in range(0, 352-96)]
            band_ind = [band_name for index, band_name in enumerate(list(cube_data.keys())) if "µm" in band_name and str(band) in band_name.split("_")[1]]
            band_ind = band_ind[0]
            # band_wave = wave_band.split("_")[1] + "_" + wave_band.split("_")[0]
            wave_data = cube_data[band_ind]
            north_slant = wave_data["north_side"]
            south_slant = wave_data["south_side"]

            # plot the cube and the lat
            # if plot_index-shift < 96:
            #     pic = cube_data["meta"]["cube_vis"]["lat"]
            #     north_slant_b = np.array([cube_vis[plot_index-shift][pixel_index[0], pixel_index[1]]
            #                              for pixel_index in north_slant["pixel_indices"]])
            #     south_slant_b = np.array([cube_vis[plot_index-shift][pixel_index[0], pixel_index[1]]
            #                              for pixel_index in south_slant["pixel_indices"]])
            # else:
            #     pic = cube_data["meta"]["cube_ir"]["lat"]
            #     north_slant_b = np.array([cube_ir[plot_index-shift-96][pixel_index[0], pixel_index[1]]
            #                              for pixel_index in north_slant["pixel_indices"]])
            #     south_slant_b = np.array([cube_ir[plot_index-shift-96][pixel_index[0], pixel_index[1]]
            #                              for pixel_index in south_slant["pixel_indices"]])
            # if not np.all(north_slant_b == north_slant["brightness_values"]):
            #     raise ValueError("Brightness values do not match")
            # if not np.all(south_slant_b == south_slant["brightness_values"]):
            #     raise ValueError("Brightness values do not match")

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
            data["time"].append(self.get_time(cube_data["meta"]["cube_vis"]["time"]))
            data["north_u1"].append(nu1)
            data["north_u2"].append(nu2)
            data["south_u1"].append(su1)
            data["south_u2"].append(su2)

            data["north u1 + u2"].append(nu1+nu2)
            data["north u1 - u2"].append(nu1-nu2)
            data["north u1 + 2*u2"].append(nu1+2*nu2)
            data["south u1 + u2"].append(su1+su2)
            data["south u1 - u2"].append(su1-su2)
            data["south u1 + 2*u2"].append(su1+2*su2)
            data["north - south u1 + u2"].append(nu1+nu2 - (su1+su2))
        lists =  list(data.keys())


        plt.plot(data["time"], data["north u1 + u2"], label="North | µ1 + µ2", linewidth = 2, color=(1, 0.78, 0.33))
        plt.plot(data["time"], data["south u1 + u2"], label="South | µ1 + µ2", linewidth = 2, color=(0.31, 0.72, 0.73))
        plt.plot(data["time"], data["north - south u1 + u2"], label="North-South | µ1 + µ2", linewidth = 2, color="red")
        plt.plot(data["time"], data["north u1 + u2"], linewidth = 2, color=(1, 0.78, 0.33))
        plt.plot(data["time"], data["south u1 + u2"], linewidth = 2, color=(0.31, 0.72, 0.73))
        plt.plot(data["time"], data["north - south u1 + u2"], linewidth = 2, color="red")
        # for index, (key, val) in enumerate(data.items()):
        #     if index == 0:
        #         plt.plot(val["wave"], val["north u1 + u2"], label="North | µ1 + µ2", linewidth = 2, color=(1, 0.78, 0.33))
        #         plt.plot(val["wave"], val["south u1 + u2"], label="South | µ1 + µ2", linewidth = 2, color=(0.31, 0.72, 0.73))
        #         plt.plot(val["wave"], val["north - south u1 + u2"], label="North-South | µ1 + µ2", linewidth = 2, color="red")
        #     else:
        #         plt.plot(val["wave"], val["north u1 + u2"], linewidth = 2, color=(1, 0.78, 0.33))
        #         plt.plot(val["wave"], val["south u1 + u2"], linewidth = 2, color=(0.31, 0.72, 0.73))
        #         plt.plot(val["wave"], val["north - south u1 + u2"], linewidth = 2, color="red")
        #     if index < len(lists) - 1:
        #         vals = data[lists[index+1]]
        #         inbetween = {"wave": (val["wave"][-1], data[list(data.keys())[index+1]]["wave"][0]),
        #                     "north u1 + u2": (val["north u1 + u2"][-1], data[list(data.keys())[index+1]]["north u1 + u2"][0]),
        #                     "north u1 + 2*u2": (val["north u1 + 2*u2"][-1], data[list(data.keys())[index+1]]["north u1 + 2*u2"][0]),
        #                     "north u1 - u2": (val["north u1 - u2"][-1], data[list(data.keys())[index+1]]["north u1 - u2"][0]),
        #                     "south u1 + u2": (val["south u1 + u2"][-1], data[list(data.keys())[index+1]]["south u1 + u2"][0]),
        #                     "south u1 + 2*u2": (val["south u1 + 2*u2"][-1], data[list(data.keys())[index+1]]["south u1 + 2*u2"][0]),
        #                     "south u1 - u2": (val["south u1 - u2"][-1], data[list(data.keys())[index+1]]["south u1 - u2"][0]),
        #                     "north - south u1 + u2": (val["north - south u1 + u2"][-1], data[list(data.keys())[index+1]]["north - south u1 + u2"][0])
        #                     }
        #         plt.plot(inbetween["wave"], inbetween["north u1 + u2"], color=(0.5, 0.39, 0.165), linestyle = "--", linewidth = 1)
        #         plt.plot(inbetween["wave"], inbetween["south u1 + u2"], color=(0.15, 0.36, 0.365), linestyle = "--", linewidth = 1)
        #         plt.plot(inbetween["wave"], inbetween["north - south u1 + u2"],  color=(0.5,0,0), linestyle = "--", linewidth = 1)
        # plt.xticks(np.arange(0.25, 3.3, 0.25),  labels=[
        #            str(x) + "µm" for x in np.arange(0.25, 3.3, 0.25)])
        # plt.xticks(np.arange(0.25, 3.3, 0.0625), minor=True)
        # find the min and max of the y axis
        ymin = np.nanmin((data["north u1 + u2"], data["south u1 + u2"],
                      data["north - south u1 + u2"]))
        ymax = np.nanmax((data["north u1 + u2"], data["south u1 + u2"],
                      data["north - south u1 + u2"]))
        # round ymin down to the nearest 0.25
        ymin -= ymin % 0.25
        ymax -= ymax % 0.25 - 0.25
        # plt.xlim(0.25, 3.25)
        plt.ylim(ymin, ymax)
        plt.yticks(np.arange(ymin, ymax + 0.001, 0.25))
        plt.yticks(np.arange(ymin, ymax + 0.001, 0.0625), minor=True)
        # plt.plot(sorted_wvlng, sorted_s_two_plus, label = "south u1 + 2*u2")
        plt.axhline(y=0, color='black', linestyle='--')
        plt.text(0.5, 0.98, "Limb Darkening", transform=plt.gca().transAxes,
                 ha='center', va='top', fontsize=18)
        plt.text(0.5, 0.05, "Limb Brightening", transform=plt.gca().transAxes,
                 ha='center', va='top', fontsize=18)

        plt.xlabel("Time")
        plt.ylabel("µ1 + µ2", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # plt.show()

        print("Saving figure to", path)
        fig.savefig(path, dpi=150)
        # plt.show()
        plt.close()

    def gen_u_vs_time(self, multi_process: Union[bool,int] = False, data=None):
        # SETTINGS["processing"]["redo_figures"])
        if data == None:
            data = self.selected_data
        args = []
        force_write = self.cache or SETTINGS["processing"]["redo_u_vs_time_per_wavelength_figure_generation"]
        bands_to_look_at = [10, 60, 155]
        for band in bands_to_look_at:
            self.u_vs_time_averages(data, band, force_write)
        for band in range(1,352+1):
            # if band != 119:
            #     continue
            if multi_process:
                args.append([data, band, force_write])
            else:
                self.u_vs_time(data, band, force_write)
                
        if multi_process == True or multi_process >= 1:
            args =[]
        
        if multi_process == True:
            multi_process_core_count = 3 # default val
        elif type(multi_process) == int:
            multi_process_core_count = multi_process
        if multi_process_core_count == 1:
            multi_process = False


        if multi_process:
            with multiprocessing.Pool(processes=multi_process_core_count) as pool:
                pool.starmap(self.u_vs_time, args)
    def u1_u2_all_figures(self, multi_process: bool = False):

        # or SETTINGS["processing"]["redo_figures"])
        # self.cube_count = len(data)
        self.gen_u_vs_time(multi_process=multi_process)
        self.gen_u_vs_wavelength(multi_process=multi_process)
        self.start_time = time.time()
        
