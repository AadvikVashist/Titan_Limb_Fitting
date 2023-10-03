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

import multiprocessing


def save_fig(fig, save_path):
    fig.savefig(save_path, dpi=150)


class gen_plots:
    def __init__(self, devEnvironment: bool = True):
        if devEnvironment == True:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["dev_figures_sub_path"],  SETTINGS["paths"]["figure_subpath"]["image_overlay"])
        else:
            self.dps_dir = join_strings(
                SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["prod_figures_sub_path"],  "DPS_55")

            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["prod_figures_sub_path"],  SETTINGS["paths"]["figure_subpath"]["image_overlay"])
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
    def gen_image_overlay(self, cube_name: str = None, band : int = None, legend_location : str = "best"):        
        if cube_name == None:
            raise ValueError("cube_name cannot be None")
        elif band == None:
            raise ValueError("band cannot be None")
        center_color = (1,0,0)
        lowest_inc_color = (8/255,130/255,12/255)
        north_color = (1,1,1)
        south_color = (0.7,0.7,0.7)
        north_slant_color = (66/255, 126/255, 167/255)
        south_slant_color = (226/255,148/255,32/255)


        data = self.get_data()
        try:    
            cube = data[cube_name]
        except:
            raise ValueError("cube_name not found in data")
        bands = list(cube.keys())
        band_ind = [band_name for index, band_name in enumerate(bands) if "µm" in band_name and str(band) in band_name.split("_")[1]]
        band_data = cube[band_ind[0]]
        if band > 96:
            image = cube["meta"]["cube_ir"]["bands"][band-97]
            center_of_cube = cube["meta"]["center_of_cube_ir"]
            north_orientation = cube["meta"]["north_orientation_ir"]
            lowest_inc = cube["meta"]["lowest_inc_location_ir"]
        else:
            image = cube["meta"]["cube_vis"]["bands"][band-1]
            center_of_cube = cube["meta"]["center_of_cube_vis"]
            north_orientation = cube["meta"]["north_orientation_vis"]
            lowest_inc = cube["meta"]["lowest_inc_location_vis"]
        fig = plt.figure(figsize=(7.5, 7.5))
        # plt.title(cube_name + " at " + band_ind[0].split("_")[0])
        plt.imshow(image, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        north_slant = band_data["north_side"]
        south_slant = band_data["south_side"]
        shape = cube["meta"]["cube_ir"]["lat"].shape
        # plt.plot([center_of_cube[1], center_of_cube[1] + np.sin(np.radians(north_orientation)) * 50],
        #             [center_of_cube[0], center_of_cube[0] - np.cos(np.radians(north_orientation)) * 50], color= north_color, label="North", linewidth=3)
        
        # plt.plot([center_of_cube[1], center_of_cube[1] - np.sin(np.radians(north_orientation)) * 50],
        #             [center_of_cube[0], center_of_cube[0] + np.cos(np.radians(north_orientation)) * 50], color= south_color, label="South", linewidth=1)

        plt.scatter(center_of_cube[1], center_of_cube[0], color=center_color, s = 120 , label="Center of Disk")
        plt.scatter(lowest_inc[1], lowest_inc[0], color= lowest_inc_color, s = 120 , label="Lowest Incidence Angle")
            
        plt.xlim(0, shape[1]-1)
        plt.ylim(shape[0]-1, 0)
        

        
        #put the slant data on to the lat array 
        area = len([dat for dat in cube["meta"]["cube_vis"]["ground"].flatten() if dat != False])
        length = np.sqrt(area / np.pi) * 1.15

        plt.plot([center_of_cube[1], center_of_cube[1] + np.sin(np.radians(north_orientation + north_slant["angle"])) * length],
                    [center_of_cube[0], center_of_cube[0] - np.cos(np.radians(north_orientation + north_slant["angle"])) * length], color=north_slant_color, label="Northern Transect", linestyle="dashed", linewidth=6)  
        plt.plot([center_of_cube[1], center_of_cube[1] + np.sin(np.radians(north_orientation + south_slant["angle"])) * length],
                    [center_of_cube[0], center_of_cube[0] - np.cos(np.radians(north_orientation + south_slant["angle"])) * length], color=south_slant_color, label="Southern Transect", linestyle="dashed", linewidth=6)  
        

        
        plt.legend(loc=legend_location, fontsize=20, facecolor =  "white", framealpha = 1)
        plt.tight_layout()
        plt.show()



    def gen_image_overlay_dps(self, cube_name: str = None, band : int = None, legend_location : str = "best"):        
        if cube_name == None:
            raise ValueError("cube_name cannot be None")
        elif band == None:
            raise ValueError("band cannot be None")
        center_color = (1,0,0)
        lowest_inc_color = (8/255,130/255,12/255)
        north_color = (1,1,1)
        south_color = (0.7,0.7,0.7)
        north_slant_color = (66/255, 126/255, 167/255)
        south_slant_color = (226/255,148/255,32/255)



        fig, ax= plt.subplots()

        plt.plot([0,0],[0,0], color=north_slant_color, label="Northern Transect", linestyle="dashed", linewidth=6)  
        plt.plot([0,0],[0,0], color=south_slant_color, label="Southern Transect", linestyle="dashed", linewidth=6)  

        plt.scatter((0),(0), s = 120 , color = center_color, label="Center of Disk")
        plt.scatter((0),(0), color= lowest_inc_color, s = 120 , label="Lowest Incidence Angle")
            
        plt.xlim(2,3)
        plt.ylim(2,3)
        plt.xticks([])
        plt.yticks([])
        plt.legend(fontsize = 20, facecolor =  "white", framealpha = 1, frameon = False, loc = "center")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)   
        fig.tight_layout(pad= 0)
        path = join_strings(self.dps_dir,  "image_legend.png")
        fig.savefig(path, dpi=300)
        print("Saved to", path)
        plt.close()
        
        
        
        data = self.get_data()
        try:    
            cube = data[cube_name]
        except:
            raise ValueError("cube_name not found in data")
        bands = list(cube.keys())
        band_ind = [band_name for index, band_name in enumerate(bands) if "µm" in band_name and str(band) in band_name.split("_")[1]]
        band_data = cube[band_ind[0]]
        if band > 96:
            image = cube["meta"]["cube_ir"]["bands"][band-97]
            center_of_cube = cube["meta"]["center_of_cube_ir"]
            north_orientation = cube["meta"]["north_orientation_ir"]
            lowest_inc = cube["meta"]["lowest_inc_location_ir"]
            shape = cube["meta"]["cube_ir"]["lat"].shape

        else:
            image = cube["meta"]["cube_vis"]["bands"][band-1]
            center_of_cube = cube["meta"]["center_of_cube_vis"]
            north_orientation = cube["meta"]["north_orientation_vis"]
            lowest_inc = cube["meta"]["lowest_inc_location_vis"]
            shape = cube["meta"]["cube_vis"]["lat"].shape

        fig = plt.figure(figsize=(7.5, 7.5))
        # plt.title(cube_name + " at " + band_ind[0].split("_")[0])
        plt.imshow(image, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        north_slant = band_data["north_side"]
        south_slant = band_data["south_side"]
        # plt.plot([center_of_cube[1], center_of_cube[1] + np.sin(np.radians(north_orientation)) * 50],
        #             [center_of_cube[0], center_of_cube[0] - np.cos(np.radians(north_orientation)) * 50], color= north_color, label="North", linewidth=3)
        
        # plt.plot([center_of_cube[1], center_of_cube[1] - np.sin(np.radians(north_orientation)) * 50],
        #             [center_of_cube[0], center_of_cube[0] + np.cos(np.radians(north_orientation)) * 50], color= south_color, label="South", linewidth=1)

        plt.scatter(center_of_cube[1], center_of_cube[0], color=center_color, s = 120 , label="Center of Disk")
        plt.scatter(lowest_inc[1], lowest_inc[0], color= lowest_inc_color, s = 120 , label="Lowest Incidence Angle")
            
        plt.xlim(0, shape[1]-1)
        plt.ylim(shape[0]-1, 0)
        

        
        #put the slant data on to the lat array 
        area = len([dat for dat in cube["meta"]["cube_vis"]["ground"].flatten() if dat != False])
        length = np.sqrt(area / np.pi) * 1.15

        plt.plot([center_of_cube[1], center_of_cube[1] + np.sin(np.radians(north_orientation + north_slant["angle"])) * length],
                    [center_of_cube[0], center_of_cube[0] - np.cos(np.radians(north_orientation + north_slant["angle"])) * length], color=north_slant_color, label="Northern Transect", linestyle="dashed", linewidth=6)  
        plt.plot([center_of_cube[1], center_of_cube[1] + np.sin(np.radians(north_orientation + south_slant["angle"])) * length],
                    [center_of_cube[0], center_of_cube[0] - np.cos(np.radians(north_orientation + south_slant["angle"])) * length], color=south_slant_color, label="Southern Transect", linestyle="dashed", linewidth=6)  
        

        
        # plt.legend(loc=legend_location, fontsize=20, facecolor =  "white", framealpha = 1)
        plt.tight_layout(pad = 0)
        band_wave = band_ind[0].split("_")[1] + "_" + band_ind[0].split("_")[0]
        path = join_strings(self.dps_dir,  "image_overlay_" + cube_name + "_" + band_wave + ".png")
        fig.savefig(path, dpi=500)
        print("Saved to", path)
        plt.close()
        fig = plt.figure(figsize=(7.5, 7.5))
        plt.imshow(image, cmap="gray")
        plt.xticks([])
        plt.yticks([])
        shape = cube["meta"]["cube_ir"]["lat"].shape
        plt.xlim(0, shape[1]-1)
        plt.ylim(shape[0]-1, 0)
        plt.tight_layout(pad = 0)
        path = join_strings(self.dps_dir,  "image_" + cube_name + "_" + band_wave + ".png")
        fig.savefig(path, dpi=500)
        print("Saved to", path)
        plt.close()
        