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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
import cv2
import multiprocessing
from .misc_plotting import gen_plots

def save_fig(fig, save_path):
    fig.savefig(save_path, dpi=150)


class gen_quad_plots:
    def __init__(self, devEnvironment: bool = True):
        if devEnvironment == True:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["dev_figures_sub_path"],  SETTINGS["paths"]["figure_subpath"]["quadrant"])
        else:
            self.dps_dir = join_strings(
                SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["prod_figures_sub_path"],  "DPS_55")
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["prod_figures_sub_path"],  SETTINGS["paths"]["figure_subpath"]["quadrant"])
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
        if os.path.exists(join_strings(self.selective_fitted_path, get_cumulative_filename("selected_sub_path"))):
            all_data = check_if_exists_or_write(join_strings(
                self.selective_fitted_path, get_cumulative_filename("selected_sub_path")), save=False, verbose=True)
        else:
            cubs = os.listdir(self.selective_fitted_path)
            cubs.sort()
            cubs = [cub for cub in cubs if re.fullmatch(
                r'C.*_.*\.pkl', cub) is not None]
            for cub in cubs:
                cube_name = os.path.splitext(cub)[0]
                all_data[cube_name] = check_if_exists_or_write(
                    join_strings(self.selective_fitted_path, cub), save=False, verbose=True)
        all_data =  dict(sorted(all_data.items()))
        return all_data
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
        mpl.rcParams['path.simplify_threshold'] = 1.0
        mpl.style.use('fast')
        plt.rcParams['font.family'] = 'serif'
        fit_obj = fit_data()
        shift = 0
        
        
        
        center_color = (1,0,0)
        lowest_inc_color = (8/255,130/255,12/255)
        north_color = (1,1,1)
        south_color = (0.7,0.7,0.7)
        north_slant_color = (66/255, 126/255, 167/255)
        south_slant_color = (226/255,148/255,32/255)


        for plot_index, (wave_band, wave_data) in enumerate(data.items()):
            if "µm_" not in wave_band:
                shift +=1
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

            #get the slant data
            north_slant = wave_data["north_side"]
            south_slant = wave_data["south_side"]
            #plot the cube and the lat
            if plot_index-shift < 96:
                suffix = "_vis"
                axs[0].imshow(cube_vis[plot_index-shift], cmap="gray")
                pic = data["meta"]["cube_vis"]["lat"]
                north_slant_b = np.array([cube_vis[plot_index-shift][pixel_index[0], pixel_index[1]] for pixel_index in north_slant["pixel_indices"]])
                south_slant_b = np.array([cube_vis[plot_index-shift][pixel_index[0], pixel_index[1]] for pixel_index in south_slant["pixel_indices"]])
            else:
                suffix = "_ir"
                axs[0].imshow(cube_ir[plot_index-96-shift], cmap="gray")
                pic =  data["meta"]["cube_ir"]["lat"]
                north_slant_b = np.array([cube_ir[plot_index-shift-96][pixel_index[0], pixel_index[1]] for pixel_index in north_slant["pixel_indices"]])
                south_slant_b = np.array([cube_ir[plot_index-shift-96][pixel_index[0], pixel_index[1]] for pixel_index in south_slant["pixel_indices"]])
            if not np.all(north_slant_b == north_slant["brightness_values"]):
                raise ValueError("Brightness values do not match")
            if not np.all(south_slant_b == south_slant["brightness_values"]):
                raise ValueError("Brightness values do not match")
            for pixel_index in north_slant["pixel_indices"]:
                pic[pixel_index[0], pixel_index[1]] = -90
            for pixel_index in south_slant["pixel_indices"]:
                pic[pixel_index[0], pixel_index[1]] = -90               
            im = axs[1].imshow(pic, cmap="gray")
            shape = data["meta"]["cube_ir"]["lat"].shape
            axs[1].plot([data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][1] + np.sin(np.radians(data["meta"]["north_orientation" + suffix])) * 50],
                        [data["meta"]["center_of_cube" + suffix][0], data["meta"]["center_of_cube" + suffix][0] - np.cos(np.radians(data["meta"]["north_orientation" + suffix])) * 50], color= north_color, label="North", linewidth=1)
            
            axs[1].plot([data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][1] - np.sin(np.radians(data["meta"]["north_orientation" + suffix])) * 50],
                        [data["meta"]["center_of_cube" + suffix][0], data["meta"]["center_of_cube" + suffix][0] + np.cos(np.radians(data["meta"]["north_orientation" + suffix])) * 50], color= south_color, label="North", linewidth=1)      

            axs[1].scatter(data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][0], color=center_color, s = 60 , label="Center of Disk")
            axs[1].scatter(data["meta"]["lowest_inc_location" + suffix][1], data["meta"]["lowest_inc_location" + suffix][0], color=lowest_inc_color, s = 60 , label="Lowest Incidence Point")

            axs[0].set_title(cube_name + " " + wave_band, fontsize=16)
            axs[1].set_title(cube_name + " Latitudes", fontsize=16)
            
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            
            axs[1].set_xlim(0, shape[1]-1)
            axs[1].set_ylim(shape[0]-1, 0)
            

            
            #put the slant data on to the lat array 
            area = len([dat for dat in data["meta"]["cube_vis"]["ground"].flatten() if dat != False])
            length = np.sqrt(area / np.pi) 

            axs[1].plot([data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][1] + np.sin(np.radians(data["meta"]["north_orientation" + suffix] + north_slant["angle"])) * length],
                        [data["meta"]["center_of_cube" + suffix][0], data["meta"]["center_of_cube" + suffix][0] - np.cos(np.radians(data["meta"]["north_orientation" + suffix] + north_slant["angle"])) * length], color=north_slant_color, label="Northern Facing Slant", linestyle="dashed" ) 
            axs[1].plot([data["meta"]["center_of_cube" + suffix][1], data["meta"]["center_of_cube" + suffix][1] + np.sin(np.radians(data["meta"]["north_orientation" + suffix] + south_slant["angle"])) * length],
                        [data["meta"]["center_of_cube" + suffix][0], data["meta"]["center_of_cube" + suffix][0] - np.cos(np.radians(data["meta"]["north_orientation" + suffix] + south_slant["angle"])) * length],color=south_slant_color, label="South Facing Slant", linestyle="dashed", )
            axs[1].legend(loc="upper left")



            #start bottom row of plots 
            
            axs[2].set_title(str(north_slant["angle"]) + "° relative to north", fontsize=16)
            axs[3].set_title(str(south_slant["angle"]) + "° relative to north", fontsize=16)


            #figure out what the best y range is for the plot
            range_vals = list(north_slant["brightness_values"]); range_vals.extend(south_slant["brightness_values"])
            y_range = np.max(range_vals) - np.min(range_vals)
            y_min = np.min(range_vals) - y_range * 0.1; y_min -= y_min % 0.01
            y_max = np.max(range_vals) + y_range * 0.1; y_max += y_max % 0.01
            
            if y_range < 0.05:
                axs[2].set_yticks(np.arange(y_min,y_max+0.001,0.0025), minor=True)
                axs[3].set_yticks(np.arange(y_min,y_max+0.001,0.0025), minor=True)
                
            #set plot styles
            
            axs[2].set_xlabel("Normalized Distance (from center of disk)", fontsize=12)
            axs[2].set_ylabel("Brightness Value", fontsize=14)
            axs[2].set_ylim(y_min,y_max)
            axs[2].set_xlim(0,1)
            axs[2].set_yticks(np.arange(y_min,y_max+0.001,0.01), labels = np.around(np.arange(y_min,y_max+0.001,0.01), 3), fontsize=14)
            axs[2].set_xticks(np.arange(0,1.1,0.1), labels = np.around(np.arange(0,1.1,0.1), 3), fontsize=14)
            axs[2].set_xticks(np.arange(0,1.1,0.05), minor=True)
            
            axs[3].set_xlabel("Normalized Distance (from center of disk)", fontsize=12)
            axs[3].set_ylabel("Brightness Value", fontsize=14)
            axs[3].set_ylim(y_min,y_max)

            axs[3].set_xlim(0,1)
            axs[3].set_yticks(np.arange(y_min,y_max+0.001,0.01), labels = np.around(np.arange(y_min, y_max+0.001, 0.01),3),  fontsize=14)
            axs[3].set_xticks(np.arange(0,1.1,0.1), labels = np.around(np.arange(0,1.1,0.1), 3),  fontsize=14)
            axs[3].set_xticks(np.arange(0,1.1,0.05), minor=True)
            


            #plot the data
            normalized_distances = self.emission_to_normalized(emission_angle=north_slant["emission_angles"])
            axs[2].plot(1 - normalized_distances, north_slant["brightness_values"], color= (0.2,0.6,0.3), label="north_side_profile")



            #work with fits
            if north_slant["meta"]["processing"]["fitted"] == True and len(north_slant["fit"]["quadratic"]["optimal_fit"]) != 0:
                interp_distances = np.linspace(np.min(normalized_distances),np.max(normalized_distances),50)
                fitted_values = fit_obj.quadratic_limb_darkening(interp_distances,*list(north_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].values()))
                axs[2].plot(1 - interp_distances, fitted_values, color= (0.9,0.2,0.2), label = "fit")
                string = '\n'.join([str(key) + " : " + str(np.around(item,4)) for key,item in north_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].items()])  + "\n r2_score: " + str(np.around(north_slant["fit"]["quadratic"]["optimal_fit"]["r2"],4))
                axs[2].text(0.95, 0.03, string,
                        verticalalignment='bottom', horizontalalignment='right',
                        transform=axs[2].transAxes,
                        color='black', fontsize=10,
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
            
            normalized_distances = self.emission_to_normalized(emission_angle=south_slant["emission_angles"])
            axs[3].plot(1 - normalized_distances, south_slant["brightness_values"], color=(0.2,0.5,0.6), label="south_side_profile")

            
            if south_slant["meta"]["processing"]["fitted"] == True  and len(south_slant["fit"]["quadratic"]["optimal_fit"]) != 0:
                interp_distances = np.linspace(np.min(normalized_distances),np.max(normalized_distances),50)
                fitted_values = fit_obj.quadratic_limb_darkening(interp_distances,*list(south_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].values()))
                axs[3].plot(1 - interp_distances, fitted_values,color= (0.9,0.2,0.2), label = "fit")
                string = '\n'.join([str(key) + " : " +str(np.around(item,4)) for key,item in south_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].items()])  + "\n r2_score: " + str(np.around(south_slant["fit"]["quadratic"]["optimal_fit"]["r2"],4))
                axs[3].text(0.95, 0.03, string,
                        verticalalignment='bottom', horizontalalignment='right',
                        transform=axs[3].transAxes,
                        color='black', fontsize=10,
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
            

            axs[2].legend()
            axs[2].grid(True, which='both', axis='both', linestyle='--')

            axs[3].legend()
            axs[3].grid(True, which='both', axis='both', linestyle='--')
            fig.tight_layout()

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




    def gen_cube_quad_dps(self, data: dict, cube_index: int, cube_name: str = None,band_value = 115):
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
        if not os.path.exists(self.dps_dir):
            os.makedirs(self.dps_dir)
        # cube_vis = pyvims.VIMS(cube_name + "_vis.cub", join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube_name), channel="vis")
        # cube_ir = pyvims.VIMS(cube_name + "_ir.cub",  join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube_name), channel="ir")

        cube_vis = [data["meta"]["cube_vis"]["bands"][index]
                    for index in range(0, 96)]
        cube_ir = [data["meta"]["cube_ir"]["bands"][index]
                   for index in range(0, 352-96)]
        mpl.rcParams['path.simplify_threshold'] = 1.0
        mpl.style.use('fast')
        # plt.rcParams['font.family'] = 'serif'
        fit_obj = fit_data()
        shift = 0
        
        

        north_slant_color =  np.array((96,163,208))/255
        south_slant_color = (226/255,148/255,32/255)
        fit_color = np.array((67,97,0))/255
        x = gen_plots(devEnvironment=self.devEnvironment)
        x.gen_image_overlay_dps(cube_name, band_value)
        for plot_index, (wave_band, wave_data) in enumerate(data.items()):
            if "µm_" not in wave_band:
                shift +=1
                continue
            if int(wave_band.split("_")[1]) != band_value:
                continue
            band_wave = wave_band.split("_")[1] + "_" + wave_band.split("_")[0]


            fig, axs = plt.subplots(1, 2, figsize=(12, 8))
            axs = axs.flatten()

            #get the slant data
            north_slant = wave_data["north_side"]
            south_slant = wave_data["south_side"]
            #plot the cube and the lat
            if plot_index-shift < 96:
                suffix = "_vis"
                pic = cube_vis[plot_index-shift]
                north_slant_b = np.array([cube_vis[plot_index-shift][pixel_index[0], pixel_index[1]] for pixel_index in north_slant["pixel_indices"]])
                south_slant_b = np.array([cube_vis[plot_index-shift][pixel_index[0], pixel_index[1]] for pixel_index in south_slant["pixel_indices"]])
            else:
                suffix = "_ir"
                pic = cube_ir[plot_index-96-shift]
                north_slant_b = np.array([cube_ir[plot_index-shift-96][pixel_index[0], pixel_index[1]] for pixel_index in north_slant["pixel_indices"]])
                south_slant_b = np.array([cube_ir[plot_index-shift-96][pixel_index[0], pixel_index[1]] for pixel_index in south_slant["pixel_indices"]])
            if not np.all(north_slant_b == north_slant["brightness_values"]):
                raise ValueError("Brightness values do not match")
            if not np.all(south_slant_b == south_slant["brightness_values"]):
                raise ValueError("Brightness values do not match")
            for pixel_index in north_slant["pixel_indices"]:
                pic[pixel_index[0], pixel_index[1]] = 0
            for pixel_index in south_slant["pixel_indices"]:
                pic[pixel_index[0], pixel_index[1]] = 0  
            #start bottom row of plots 
            
            axs[0].set_title(str(north_slant["angle"]) + "° relative to north", fontsize=16)
            axs[1].set_title(str(south_slant["angle"]) + "° relative to north", fontsize=16)


            #figure out what the best y range is for the plot
            range_vals = list(north_slant["brightness_values"]); range_vals.extend(south_slant["brightness_values"])
            y_range = np.max(range_vals) - np.min(range_vals)
            y_min = np.min(range_vals) - y_range * 0.1; y_min -= y_min % 0.01
            y_max = np.max(range_vals) + y_range * 0.1; y_max += y_max % 0.01
            
            if y_range < 0.05:
                axs[0].set_yticks(np.arange(y_min,y_max+0.001,0.0025), minor=True)
                axs[1].set_yticks(np.arange(y_min,y_max+0.001,0.0025), minor=True)
                
            #set plot styles
            
            axs[0].set_xlabel("Normalized Distance (from center of disk)", fontsize=12)
            axs[0].set_ylabel("Brightness Value", fontsize=14)
            axs[0].set_ylim(y_min,y_max)
            axs[0].set_xlim(0,1)
            axs[0].set_yticks(np.arange(y_min,y_max+0.001,0.01), labels = np.around(np.arange(y_min,y_max+0.001,0.01), 3), fontsize=14)
            axs[0].set_xticks(np.arange(0,1.1,0.1), labels = np.around(np.arange(0,1.1,0.1), 3), fontsize=14)
            axs[0].set_xticks(np.arange(0,1.1,0.05), minor=True)
            
            axs[1].set_xlabel("Normalized Distance (from center of disk)", fontsize=12)
            axs[1].set_ylabel("Brightness Value", fontsize=14)
            axs[1].set_ylim(y_min,y_max)

            axs[1].set_xlim(0,1)
            axs[1].set_yticks(np.arange(y_min,y_max+0.001,0.01), labels = np.around(np.arange(y_min, y_max+0.001, 0.01),3),  fontsize=14)
            axs[1].set_xticks(np.arange(0,1.1,0.1), labels = np.around(np.arange(0,1.1,0.1), 3),  fontsize=14)
            axs[1].set_xticks(np.arange(0,1.1,0.05), minor=True)
            


            #plot the data
            normalized_distances = self.emission_to_normalized(emission_angle=north_slant["emission_angles"])
            axs[0].plot(1 - normalized_distances, north_slant["brightness_values"], color= north_slant_color, label="Northern Transect", linewidth=5)



            #work with fits
            if north_slant["meta"]["processing"]["fitted"] == True and len(north_slant["fit"]["quadratic"]["optimal_fit"]) != 0:
                interp_distances = np.linspace(np.min(normalized_distances),np.max(normalized_distances),50)
                fitted_values = fit_obj.quadratic_limb_darkening(interp_distances,*list(north_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].values()))
                axs[0].plot(1 - interp_distances, fitted_values, color= fit_color, label = "Quadratic Best Fit", linestyle="dashed", linewidth=2)
                string = '\n'.join([str(key) + " : " + str(np.around(item,4)) for key,item in north_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].items()])  + "\n r2_score: " + str(np.around(north_slant["fit"]["quadratic"]["optimal_fit"]["r2"],4))
                axs[0].text(0.95, 0.03, string,
                        verticalalignment='bottom', horizontalalignment='right',
                        transform=axs[0].transAxes,
                        color='black', fontsize=10,
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
            
            normalized_distances = self.emission_to_normalized(emission_angle=south_slant["emission_angles"])
            axs[1].plot(1 - normalized_distances, south_slant["brightness_values"], color= south_slant_color, label="Southern Transect", linewidth=5)

            
            if south_slant["meta"]["processing"]["fitted"] == True  and len(south_slant["fit"]["quadratic"]["optimal_fit"]) != 0:
                interp_distances = np.linspace(np.min(normalized_distances),np.max(normalized_distances),50)
                fitted_values = fit_obj.quadratic_limb_darkening(interp_distances,*list(south_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].values()))
                axs[1].plot(1 - interp_distances, fitted_values,color= fit_color, label = "Quadratic Best Fit", linestyle="dashed", linewidth=2)
                string = '\n'.join([str(key) + " : " +str(np.around(item,4)) for key,item in south_slant["fit"]["quadratic"]["optimal_fit"]["fit_params"].items()])  + "\n r2_score: " + str(np.around(south_slant["fit"]["quadratic"]["optimal_fit"]["r2"],4))
                axs[1].text(0.95, 0.03, string,
                        verticalalignment='bottom', horizontalalignment='right',
                        transform=axs[1].transAxes,
                        color='black', fontsize=10,
                        bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 5})
            

            axs[0].legend(fontsize = 14)
            axs[0].grid(True, which='both', axis='both', linestyle='--')

            axs[1].legend(fontsize = 14)
            axs[1].grid(True, which='both', axis='both', linestyle='--')
            fig.tight_layout()
            fig.subplots_adjust(wspace=0.2)

            

            path = join_strings(self.dps_dir, "plots_" + cube_name + "_" + band_wave + ".png")
            fig.savefig(path, dpi=500)
            print("Saved to", path)
            plt.close()
            
    def quad_dps(self, cube_name_key: str = None):
        data = self.get_data()
        for index, (cube_name, cube_data) in enumerate(data.items()):
            if cube_name != cube_name_key: 
                continue
            self.gen_cube_quad_dps(cube_data, index, cube_name)

    def quad_all(self, multi_process: bool = False):
        data = self.get_data()
        self.force_write = (SETTINGS["processing"]["clear_cache"]
                            or SETTINGS["processing"]["redo_quad_figure_generation"])
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
            with multiprocessing.Pool(processes=3) as pool:
                pool.starmap(self.gen_cube_quads, args)
