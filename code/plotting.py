from data_processing.fitting import fit_data
from get_settings import join_strings, check_if_exists_or_write, SETTINGS
import re
import os
import numpy as np
import datetime
from scipy.optimize import curve_fit
# area for good figures and random figures
import matplotlib.pyplot as plt
import matplotlib
import pyvims

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
        data = dict(sorted(data.items()))
        cube = data[list(data.keys())[cube_index]]
        band = cube[list(cube.keys())[band]]
        fit_obj = fit_data(fit_type="quad")
        for slant, slant_data in band.items():
            
            x = fit_obj.emission_to_normalized(np.linspace(np.min(slant_data["emission_angles"]), np.max(slant_data["emission_angles"]), 100))
            # fit_obj.I_0 = slant_data["fit"]["fit_params"]["I_0"]
            
            y = [fit_obj.quadratic_limb_darkening(xs,slant_data["fit"]["fit_params"]["I_0"],  slant_data["fit"]["fit_params"]["u1"], slant_data["fit"]["fit_params"]["u2"]) for xs in x]
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
    def coeff_vs_deg(self, band: int = None):
        data = self.get_fitted_data()
        data = dict(sorted(data.items()))
        fig, axs = plt.subplots(4, 3, figsize=(12, 8))
        axs = axs.flatten()
        
        handles = []  # To store legend handles
        for index, (cube, cube_fits) in enumerate(data.items()):
            key = [ke for ke in cube_fits.keys() if ke.split("_")[1] == str(band)][0]
            wavelength_fit = cube_fits[key]
            u_plus = []
            u_minus = []
            axs[index].set_title(cube)
            axs[index].minorticks_on()
            axs[index].set_xticks(np.arange(0,361, 60))
            axs[index].set_yticks(np.arange(-2,3.1, 1, dtype=int))
            # axs[index].set_ylabel("N/")
            axs[index].set_xlim(-20,380)
            axs[index].set_ylim(-1.5,3)
            for slant, slant_data in wavelength_fit.items():
                u1 = axs[index].scatter(slant, slant_data["fit"]["fit_params"]["u1"], color=(0, 0, 0))
                u2 = axs[index].scatter(slant, slant_data["fit"]["fit_params"]["u2"], color=(1, 0, 0))

                u_plus.append(slant_data["fit"]["fit_params"]["u1"] + slant_data["fit"]["fit_params"]["u2"])
                u_minus.append(slant_data["fit"]["fit_params"]["u1"] - slant_data["fit"]["fit_params"]["u2"])

            line_plus, = axs[index].plot(list(wavelength_fit.keys()), u_plus, color=(0, 1, 0))
            line_minus, = axs[index].plot(list(wavelength_fit.keys()), u_minus, color=(0, 0, 1))
            if index == 0:
                handles.append(u1)  # Add scatter plot handle to the list 
                handles.append(u2)
                handles.append(line_plus)  # Add line plot handle to the list
                handles.append(line_minus)  # Add line plot handle to the list
            elif index >= 9:
                axs[index].set_xlabel("Slant Angle (˚)")
            # Don't add labels to individual lines, we'll add them to the global legend
            # axs[index].legend([line_plus, line_minus], )

        # Create a global legend outside the loop
        fig.legend(handles, ["µ1", "µ2","µ1 + µ2", "µ1 - µ2"])
        fig.subplots_adjust(wspace=0.12, hspace=0.3, top=0.95, bottom=0.05, left=0.05, right=0.95)
        # Customize the markers in the legend
        # for legend_item in global_legend.legendHandles:
            # legend_item.set_markerfacecolor('black')  # Set the marker face color
            # legend_item.set_markeredgecolor('black')  # Set the marker edge color
            # legend_item.set_markersize(10)  # Set the marker size
                
        plt.show()

        # for index, (cube, cube_fits) in enumerate(cube.items()):
        #     key = [ke for ke in cube_fits.keys() if ke.split("_")[1] == str(band)][0]
        #     wavelength_fit = cube_fits[key]
        #     length = len(wavelength_fit)
        #     for ind, (degree, slant) in enumerate(wavelength_fit.items()):
        #         plt.plot(slant["emission_angles"], slant["brightness_values"], label = str(degree))
        #     plt.set_xlim(0, 95)
        #     plt.set_ylim(bottom = 0)
        # plt.legend(fontsize=6)
    def select_good_and_bad_bands(self):
        fitted_data = self.get_fitted_data()
        cubes = {}
        rejected =[0]
        if os.path.exists(os.curdir + "/rejected_bands.pkl"):
            rejected = check_if_exists_or_write(os.curdir + "/rejected_bands.pkl", save=False, verbose=True)
            
        for cube in fitted_data.keys():
            cube_vis = pyvims.VIMS(cube + "_vis.cub", join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube), channel="vis")
            cube_ir = pyvims.VIMS(cube + "_ir.cub",  join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube), channel="ir")
            cubes[cube] = {"vis": cube_vis, "ir": cube_ir}
        for band in range(1,353):
            if band < np.max(rejected):
                continue
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(4,3, figsize=(14,8))
            axs = axs.flatten()
            for index, (cube, data) in enumerate(cubes.items()):
                axs[index].set_title(cube)
                if band <= 96:
                    axs[index].imshow(data["vis"][band], cmap="gray")
                else:
                    axs[index].imshow(data["ir"][band], cmap="gray")
            plt.waitforbuttonpress(5)
            plt.close("all")
            while True:
                inp = input(str(band) + " Good? (y/n)")
                if "y" in inp:
                    break
                if "n" in inp:
                    rejected.append(band)
                    break
            
        check_if_exists_or_write(os.curdir + "/rejected_bands.pkl", save=True, data=rejected, force_write=True, verbose=True)
        return rejected

x = plotting_base()
# var = x.select_good_and_bad_bands()
# print(var)
for i in range(12):
    x.look_at_fits(cube_index=i,band=118)


#index 1, North is straight right
# figure_waves = np.linspace(1, 352, 4)
# for wave in figure_waves:
#     print(wave)
#     x.timeline_figure(band=int(wave))
# # print(x.get_fig_path("test", "test", "test"))


