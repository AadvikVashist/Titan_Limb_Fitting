
from settings.get_settings import join_strings, check_if_exists_or_write, SETTINGS, get_cumulative_filename
import re
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvims
from ..data_processing.fitting import fit_data
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
import cv2
import multiprocessing
from .misc_plotting import gen_plots
import io
from PIL import Image
from typing import Union
def save_fig(fig, save_path):
    fig.savefig(save_path, dpi=150)


class gen_image_bands:
    def __init__(self, devEnvironment: bool = True):
        if devEnvironment == True:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["dev_figures_sub_path"],  SETTINGS["paths"]["figure_subpath"]["cube_preview"])
        else:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["prod_figures_sub_path"],  SETTINGS["paths"]["figure_subpath"]["cube_preview"])
        self.devEnvironment = devEnvironment
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
    
    def rotate_image(self, img, rotation):
        fig,ax2 = plt.subplots()
        ax2.axis('off')  # Turn off the axis
        a = fig.canvas.get_width_height()
        ax2.imshow(img, cmap = "gray")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)

        # Open the image and convert to a numpy array
        image = Image.open(buf)
        image = image.rotate(rotation, expand=True)

        image_array = np.array(image)
        return image_array
    def rotate_fig(self, fig, ax, rotation):
        a = fig.canvas.get_width_height()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)

        # Open the image and convert to a numpy array
        image = Image.open(buf)
        image = image.rotate(rotation, expand=True)

        image_array = np.array(image)
        return image_array
    def gen_cube(self, data: dict, cube_index: int, cube_name: str = None):
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

        cube_vis = [data["meta"]["cube_vis"]["bands"][index]
                    for index in range(0, 96)]
        cube_ir = [data["meta"]["cube_ir"]["bands"][index]
                   for index in range(0, 352-96)]
        bands_done = 1
        mpl.rcParams['path.simplify_threshold'] = 1.0
        mpl.style.use('fast')
        plt.rcParams['font.family'] = 'serif'
        shift = 0
        
        for plot_index, (wave_band, wave_data) in enumerate(data.items()):
            if "µm_" not in wave_band:
                shift +=1
                continue
            if plot_index in SETTINGS["figure_generation"]["unusable_bands"]:
                save_end = "_unused.png"
            else:
                bands_done += 1
                save_end = ".png"
            band_wave = wave_band.split("_")[1] + "_" + wave_band.split("_")[0]
            if os.path.exists(join_strings(self.save_dir, cube_name, band_wave + save_end)) and not self.force_write:
                continue
            fig = plt.figure(figsize=(12, 12))
            axs = fig.add_subplot(111)
            axs.set_title(cube_name +" band" + band_wave.split("_")[0] + " and " + band_wave.split("_")[1], fontsize=30)
            axs.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)



            if plot_index-shift < 96:
                suffix = "_vis"
                plt.imshow(self.rotate_image(cube_vis[plot_index-shift],data["meta"]["north_orientation" + suffix]), cmap="gray")
            else:
                suffix = "_ir"
                plt.imshow(self.rotate_image(cube_ir[plot_index-96-shift],data["meta"]["north_orientation" + suffix]), cmap="gray")
            fig.tight_layout()
            fig.savefig(join_strings(self.save_dir, cube_name, band_wave + save_end), dpi=150)
            # plt.show()
            plt.close(fig)
            time_spent = np.around(time.time() - self.cube_start_time, 3)
            percentage_completed = (bands_done) / leng
            total_time_left = time_spent / percentage_completed - time_spent

            print("Finished previews for ", wave_band, "| Spent", time_spent, "so far | expected time left:",
                  np.around(total_time_left, 2), "| total time for cube :",  np.around(total_time_left + time_spent, 3), end="\r")

        time_spent = np.around(time.time() - self.cube_start_time, 3)
        percentage_completed = (cube_index + 1) / self.cube_count
        total_time_left = time_spent / percentage_completed - time_spent
        print("Cube", cube_index + 1, "of", self.cube_count, "| Total time for cube:", time_spent, "seconds | Total Expected time left:",
              np.around(total_time_left, 2), "seconds", "| Total time spent:", np.around(time.time() - self.start_time, 3), "seconds\n")



    def gen_all(self, multi_process: Union[bool,int] = False):
        data = self.get_data()
        self.force_write = (SETTINGS["processing"]["clear_cache"]
                            or SETTINGS["processing"]["redo_simple_preview_figure_generation"])
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
            if multi_process:
                args.append([cube_data, index, cube_name])
            else:
                self.gen_cube(cube_data, index, cube_name)
        if multi_process:
            with multiprocessing.Pool(processes=multi_process_core_count) as pool:
                pool.starmap(self.gen_cube, args)
