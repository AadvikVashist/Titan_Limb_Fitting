from fitting import fit_data
from get_settings import join_strings, check_if_exists_or_write, SETTINGS
import re
import os
import numpy as np
import datetime
from scipy.optimize import curve_fit
# area for good figures and random figures
import matplotlib.pyplot as plt
import matplotlib
class plotting_base:
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
        try:
            check_if_exists_or_write(self.get_fig_path("timeline", str(band), "all") + ".png", save=True, data="sdfs", force_write=True, verbose=True)
        except:
            pass
        fig.savefig(self.get_fig_path("timeline", str(band), "all") + ".png", dpi=450)
        # plt.show()

    def look_at_fits(self, cube_index: int = None, band: int = None):
        data = self.get_fitted_data()
        cube = data[list(data.keys())[cube_index]]
        band = cube[list(cube.keys())[band]]
        fit_obj = fit_data(fit_type="quad")
        for slant, slant_data in band.items():

            x = fit_obj.emission_to_normalized(np.linspace(np.min(slant_data["emission_angles"]), np.max(slant_data["emission_angles"]), 100))
            fit_obj.I_0 = slant_data["fit"]["fit_params"]["I_0"]
            
            y = [fit_obj.quadratic_limb_darkening(xs, slant_data["fit"]["fit_params"]["u1"], slant_data["fit"]["fit_params"]["u2"]) for xs in x]
            plt.plot(x, y)
            plt.plot(fit_obj.emission_to_normalized(slant_data["emission_angles"]), slant_data["brightness_values"])
            plt.show()
        # for index, (cube, cube_fits) in enumerate(cube.items()):
        #     key = [ke for ke in cube_fits.keys() if ke.split("_")[1] == str(band)][0]
        #     wavelength_fit = cube_fits[key]
        #     length = len(wavelength_fit)
        #     for ind, (degree, slant) in enumerate(wavelength_fit.items()):
        #         plt.plot(slant["emission_angles"], slant["brightness_values"], label = str(degree))
        #     plt.set_xlim(0, 95)
        #     plt.set_ylim(bottom = 0)
        plt.legend(fontsize=6)
        # plt.title("")
x = plotting_base()

x.look_at_fits(cube_index=0, band=1)
# figure_waves = np.linspace(1, 352, 4)
# for wave in figure_waves:
#     print(wave)
#     x.timeline_figure(band=int(wave))
# # print(x.get_fig_path("test", "test", "test"))
