from polar_profile import analyze_complete_dataset
from sort_and_filter import sort_and_filter
from get_settings import join_strings, check_if_exists_or_write, SETTINGS
import re
import os
import numpy as np
import datetime
from scipy.optimize import curve_fit
# area for good figures and random figures
import matplotlib.pyplot as plt
import matplotlib
class fitting_base:
    def __init__(self, devEnvironment: bool = True):
        if devEnvironment == True:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["dev_figures_sub_path"])
        else:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["prod_figures_sub_path"])
        self.filtered_data_dir = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["sorted_sub_path"])
        self.fitted_data_dir = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["fitted_sub_path"])
        self.devEnvironment = devEnvironment

    def get_filtered_data(self):
        all_data = {}
        if os.path.exists(join_strings(self.filtered_data_dir, SETTINGS["paths"]["cumulative_sorted_path"])):
            all_data = check_if_exists_or_write(join_strings(
                self.filtered_data_dir, SETTINGS["paths"]["cumulative_sorted_path"]), save=False, verbose=True)
        else:
            cubs = os.listdir(self.filtered_data_dir)
            cubs = [cub for cub in cubs if re.fullmatch(
                r'C.*_.*\.pkl', cub) is not None]
            for cub in cubs:
                cube_name = os.path.splitext(cub)[0]
                all_data[cube_name] = check_if_exists_or_write(join_strings(
                    self.filtered_data_dir, cub), save=False, verbose=True)
        return all_data

    def get_fitted_data(self):
        all_data = {}
        if os.path.exists(join_strings(self.filtered_data_dir, SETTINGS["paths"]["cumulative_fitted_path"])):
            all_data = check_if_exists_or_write(join_strings(
                self.fitted_data_dir, SETTINGS["paths"]["cumulative_fitted_path"]), save=False, verbose=True)
        else:
            cubs = os.listdir(self.fitted_data_dir)
            cubs = [cub for cub in cubs if re.fullmatch(
                r'C.*_.*\.pkl', cub) is not None]
            for cub in cubs:
                cube_name = os.path.splitext(cub)[0]
                all_data[cube_name] = check_if_exists_or_write(
                    join_strings(self.fitted_data_dir, cub), save=False, verbose=True)
        return all_data

    def get_fig_path(self, figure_type: str, fig_name: str, cube_name: str):
        base_path = join_strings(self.save_dir, figure_type)
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

    def filter_cubes(self, data, cubes : str):
        if cubes == "all":
            return data
        else:
            filtered_data = {}
            for cube in cubes:
                filtered_data[cube] = data[cube]
            return filtered_data
    
    # def fit(self, data: dict):
    #     leng = len(data)
    #     for index, (wave_band, wave_data) in enumerate(data.items()):
    #         for slant, slant_data in wave_data.items():
    #             emission_angles = np.array(slant_data["emission_angles"])
    #             brightness_values = np.array(slant_data["brightness_values"])
    #             popt, pcov = self.limb_darkening_laws(
    #                 emission_angles, brightness_values)
    #             data[wave_band][slant]["meta"]["processing"]["fitted"] = True
    #             data[wave_band][slant]["fit"] = {
    #                 "fit_params": popt, "covariance_matrix": pcov, "limb_darkening_function": self.limb_darkening_function_name}
    #             # data[wave_band][slant]["pixel_indices"], data[wave_band][slant]["pixel_distances"], data[wave_band][slant]["emission_angles"], data[wave_band][slant]["brightness_values"]
    #         print("Finished fitting", wave_band, "| Spent", np.around(time.time() - self.cube_start_time, 3),
    #               "| expected time left:", np.around((time.time() - self.cube_start_time) * (leng - index - 1), 2), end="\r")
    #     print()
    #     return data

    # def fit_all(self):
    #     data = self.get_filtered_data()
    #     force_write = (SETTINGS["processing"]["clear_cache"]
    #                    or SETTINGS["processing"]["redo_fitting"])
    #     appended_data = False
    #     cube_count = len(data)
    #     self.start_time = time.time()

    #     for index, (cube_name, cube_data) in enumerate(data.items()):
    #         if os.path.exists(join_strings(self.save_dir, cube_name + ".pkl")) and not force_write:
    #             try:
    #                 data[cube_name] = check_if_exists_or_write(
    #                     join_strings(self.save_dir, cube_name + ".pkl"), save=False)
    #                 print("fitted data already exists. Skipping...")
    #                 continue
    #             except:
    #                 print("fitted data corrupted. Processing...")
    #         elif not force_write:
    #             appended_data = True
    #         self.cube_start_time = time.time()
    #         # only important line in this function
    #         data[cube_name] = self.fit(cube_data)

    #     # Calculate expected time left based on the average time per cube
    #         check_if_exists_or_write(join_strings(
    #             self.save_dir, cube_name + ".pkl"), data=data[cube_name], save=True, force_write=True)
    #         print("Total time for cube: ", np.around(time.time() - self.cube_start_time, 3), "seconds | Expected time left:",
    #               np.around((time.time() - self.start_time) / (index + 1) * (cube_count - index - 1), 2), "seconds")

    #     if os.path.exists(join_strings(self.save_dir, SETTINGS["paths"]["cumulative_fitted_path"])) and (not force_write or appended_data):
    #         print("Since fitted data already exists, but new data has been appended ->")
    #         check_if_exists_or_write(join_strings(
    #             self.save_dir, SETTINGS["paths"]["cumulative_fitted_path"]), data=data, save=True, force_write=True, verbose=True)
    #     else:
    #         check_if_exists_or_write(join_strings(
    #             self.save_dir, SETTINGS["paths"]["cumulative_fitted_path"]), data=data, save=True, force_write=True, verbose=True)


    def timeline_figure(self, band: int = None):
        all_fits = {}
        fig, axs = plt.subplots(4, 3, figsize=(12, 8))
        axs = axs.flatten()
        plt.title(str(band))
        data = self.get_fitted_data()
        quantity = len(data)
        cmap = matplotlib.colormaps.get_cmap('bone')
        for index, (cube, cube_fits) in enumerate(data.items()):
            key = [ke for ke in cube_fits.keys() if ke.split("_")[1] == str(band)][0]
            wavelength_fit = cube_fits[key]
            length = len(wavelength_fit)
            for ind, (degree, slant) in enumerate(wavelength_fit.items()):
                axs[index].plot(slant["emission_angles"], slant["brightness_values"], label = str(degree),  color= cmap(1.6*abs(ind/length - 0.5)))
            axs[index].set_xlim(0, 95)
            axs[index].set_ylim(bottom = 0)
            axs[index].legend(fontsize=3)
            axs[index].set_title(cube)
        fig.tight_layout()
        plt.show()
        return all_fits

    # def generate_figures(self, figures: dict = None):
    #     for band in range(1, 352, 10):
    #         # if band < 97 and surface_windows[band-1] == True:
    #         #     continue
    #         # elif band > 96 and band in ir_surface_windows:
    #         #     continue
    #         self.timeline_figure(band)


x = fitting_base()

figure_waves = np.linspace(1, 352, 4)
for wave in figure_waves:
    print(wave)
    x.timeline_figure(band=int(wave))
# print(x.get_fig_path("test", "test", "test"))
