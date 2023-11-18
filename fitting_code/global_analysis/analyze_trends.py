
from settings.get_settings import join_strings, check_if_exists_or_write, SETTINGS, get_cumulative_filename
import re
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
import pyvims
from sklearn.metrics import r2_score
import datetime
class trend_analysis:
    def __init__(self, devEnvironment: bool = True):
        self.data_dir = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["selected_sub_path"])
        # if devEnvironment == True:
        #     self.save_dir = join_strings(
        #         SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["dev_figures_sub_path"],  SETTINGS["paths"]["figure_subpath"]["transitional_period"])
        # else:
        #     self.save_dir = join_strings(
        #         SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["prod_figures_sub_path"],  SETTINGS["paths"]["figure_subpath"]["transitional_period"])
        self.devEnvironment = devEnvironment
        self.unusable_bands = SETTINGS["figure_generation"]["unusable_bands"]
    def emission_to_normalized(self, emission_angle):
        return np.cos(np.deg2rad(emission_angle))

    def get_data(self):
        all_data = {}
        if os.path.exists(join_strings(self.data_dir, get_cumulative_filename("selected_sub_path"))):
            all_data = check_if_exists_or_write(join_strings(
                self.data_dir, get_cumulative_filename("selected_sub_path")), save=False, verbose=True)
        else:
            cubs = os.listdir(self.data_dir)
            cubs = [cub for cub in cubs if re.fullmatch(
                r'C.*_.*\.pkl', cub) is not None]
            for cub in cubs:
                cube_name = os.path.splitext(cub)[0]
                all_data[cube_name] = check_if_exists_or_write(
                    join_strings(self.data_dir, cub), save=False, verbose=True)
        return all_data






    def cubic_interp_zeros(self, x, y):
        interp = PchipInterpolator(x, y)
        
        zeros = []
        x = np.linspace(np.min(x), np.max(x), 3000)
        y = interp(x)
        zeros = x[np.where(np.diff(np.sign(y)))[0]]
        zeros = [zero for zero in zeros if zero > 0.5]
        return np.mean(zeros)
    
    def detector_smoothed_comparison(self, x, y):
        smoothed_y = gaussian_filter(y, sigma=4)
        

        
        interp = PchipInterpolator(x, smoothed_y)
        zeros = []
        x = np.linspace(np.min(x), np.max(x), 3000)
        y = interp(x)
        

        zeros = x[np.where(np.diff(np.sign(y)))[0]]
        smooth_zeros = [zero for zero in zeros if zero > 0.5] # 0.5 is the minimum wavelength for the detector
        
        for index, smooth_zero in enumerate(smooth_zeros): #
            smooth_zero = np.mean([zero for zero in zeros if abs(zero-smooth_zero) < 0.2]) 
            smooth_zeros[index] = smooth_zero
        if len (smooth_zeros) < 1:
            plt.plot(x, y, label = "Northern Transect")
            plt.show()
            smooth_zeros = [np.mean(smooth_zeros)]
            if len(zeros) < 1:
                return []
            else:
                return [np.mean(zeros)]
        return smooth_zeros

    def run_transitional_detector(self, data: dict, cube_name: str = None):
        leng = len(data.keys()) - 1

        wave_bands = []
        northern_transect = []
        southern_transect = []
        for index, (wave_band, wave_data) in enumerate(data.items()):
            if "µm_" not in wave_band:
                continue
            if index in self.unusable_bands:
                continue
            try:
                northern_transect.append(wave_data["north_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u1"] + wave_data["north_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u2"])
            except:
                northern_transect.append(np.nan)
            try:
                southern_transect.append(wave_data["south_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u1"] + wave_data["south_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u2"])
            except:
                southern_transect.append(np.nan)
            wave_length = wave_band.split("_")[0][0:-2]
            wave_bands.append(float(wave_length))

        wave_bands, northern_transect, southern_transect = zip(*sorted(zip(wave_bands, northern_transect, southern_transect)))
        mask = np.isfinite(northern_transect) & np.isfinite(southern_transect)
        wave_bands = np.array(wave_bands)[mask]
        northern_transect = np.array(northern_transect)[mask]
        southern_transect = np.array(southern_transect)[mask]
        smoothed_northern_transect = gaussian_filter(northern_transect, sigma=4)
        smoothed_southern_transect = gaussian_filter(southern_transect, sigma=4)
        northern_transition = self.detector_smoothed_comparison(wave_bands, northern_transect)
        southern_transition = self.detector_smoothed_comparison(wave_bands, southern_transect)
        transects =[]

        if len(northern_transition) > 1:
            transects.extend(northern_transition)
        elif len(northern_transition) == 1:
            transects.append(northern_transition[0])
        if len(southern_transition) > 1:
            transects.extend(southern_transition)
        elif len(southern_transition) == 1:
            transects.append(southern_transition[0])
        # plt.figure(figsize=(15, 5))
        # # plt.plot(wave_bands, smoothed_northern_transect, label = "Northern Transect Smoothed")
        # # plt.plot(wave_bands, smoothed_southern_transect, label = "Southern Transect Smoothed")
        # plt.xlabel("Wavelength (µm)")
        # plt.ylabel("Measure of Limb Brightness Darkness (u1+u2)")
        # plt.plot(wave_bands, northern_transect, label = "Northern Transect")
        # plt.plot(wave_bands, southern_transect, label = "Southern Transect")
        # plt.hlines(0, np.min(wave_bands), np.max(wave_bands), linestyles="dashed", label="Zero", color="black")
        # plt.vlines(transects, np.min(northern_transect), np.max(northern_transect), linestyles="dashed", label="Transition", color="red")
        # plt.legend()
        # fig_path = join_strings(self.save_dir, cube_name + ".png")
        # plt.savefig(fig_path, dpi = 300)
        # # plt.show()
        # plt.close()
        return northern_transition, southern_transition

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
    
    
    def detect_transitional_period(self, data):
        cube_time = []
        transition_north = []
        transition_south = []
        mean_transition = []
        # if not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)
        for index, (cube_name, cube_data) in enumerate(data.items()):
            self.cube_start_time = time.time()
            # only important line in this function
            cube_time.append(self.get_time(cube_data["meta"]["cube_vis"]["time"]))

            detector = self.run_transitional_detector(cube_data, cube_name)
            
            transition_north.append(np.mean(detector[0]))
            transition_south.append(np.mean(detector[1]))
            mean_transition.append(np.mean([np.mean(detector[0]), np.mean(detector[1])]))
        cube_time, transition_north, transition_south, mean_transition = zip(*sorted(zip(cube_time, transition_north, transition_south, mean_transition)))
        cube_time, transition_north, transition_south, mean_transition
        
    def analyze_all(self):
        data = self.get_data()
        force_write = (SETTINGS["processing"]["clear_cache"]
                       or SETTINGS["processing"]["redo_selection"])
        appended_data = False
        cube_count = len(data)
        self.start_time = time.time()
        self.detect_transitional_period(data)
