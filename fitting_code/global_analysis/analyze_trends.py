from settings.get_settings import join_strings, check_if_exists_or_write, SETTINGS, get_cumulative_filename
import re
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
import pyvims
from sklearn.metrics import r2_score
import datetime
from ..data_processing.fitting import fit_data
class trend_analysis:
    def __init__(self, devEnvironment: bool = True):
        self.data_dir = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["selected_sub_path"])
        self.save_dir = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["global_sub_path"])
        self.data = self.get_data()
        self.force_write = (SETTINGS["processing"]["clear_cache"] or SETTINGS["processing"]["redo_global_fitting"])

        self.fig_paths = SETTINGS["paths"]["figure_subpath"]
        if devEnvironment == True:
            self.fig_save_dir = join_strings(
                SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["dev_figures_sub_path"])
        else:
            self.fig_save_dir = join_strings(
                SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["prod_figures_sub_path"])
        self.devEnvironment = devEnvironment
        self.unusable_bands = SETTINGS["figure_generation"]["unusable_bands"]
    def emission_to_normalized(self, emission_angle):
        return np.cos(np.deg2rad(emission_angle))
    def check_if_cube(self, cube_name):
        return re.match( r'^C\d+_\d+$' , cube_name)
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
    def save(self, full_save: bool = False):
        dat = self.data
        if not full_save:
            dat = self.data["global_analysis"]
        if self.force_write or not os.path.exists(join_strings(self.save_dir, get_cumulative_filename("global_sub_path"))):
            check_if_exists_or_write(join_strings(
                    self.save_dir, get_cumulative_filename("global_sub_path")), data = dat, save=True, verbose=True)
        else:
            print("Not saving because file already exists and force_write is False")

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
        smooth_zeros = [zero for zero in zeros if zero > 0.6] # 0.5 is the minimum wavelength for the detector
        
        for index, smooth_zero in enumerate(smooth_zeros): #
            smooth_zero = np.mean([zero for zero in zeros if abs(zero-smooth_zero) < 0.2]) 
            smooth_zeros[index] = smooth_zero
        if len (smooth_zeros) < 1:
            # plt.plot(x, y, label = "Northern Transect")
            # plt.show()
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
    
    
    def detect_transitional_period(self):
        cube_time = []
        transition_north = []
        transition_south = []
        mean_transition = []
        # if not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)
        for index, (cube_name, cube_data) in enumerate(self.data.items()):
            if not self.check_if_cube(cube_name):
                continue
            self.cube_start_time = time.time()
            # only important line in this function
            cube_time.append(self.get_time(cube_data["meta"]["cube_vis"]["time"]))

            detector = self.run_transitional_detector(cube_data, cube_name)
            
            transition_north.append(np.mean(detector[0]))
            transition_south.append(np.mean(detector[1]))
            mean_transition.append(np.mean([np.mean(detector[0]), np.mean(detector[1])]))
        cube_time, transition_north, transition_south, mean_transition = zip(*sorted(zip(cube_time, transition_north, transition_south, mean_transition)))
        cube_time, transition_north, transition_south, mean_transition
        #chekc if global analysis exists
        if "global_analysis" not in self.data:
            self.data["global_analysis"] = {}
        self.data["global_analysis"]["transitional period"] = {"cube_time": cube_time, "transition_north": transition_north, "transition_south": transition_south, "mean_transition": mean_transition}
        self.save(full_save = False)
    
    def plot_transitional_period(self):
        if "global_analysis" not in self.data:
            self.data["global_analysis"] = {}
        if "transitional period" not in self.data["global_analysis"]:
            self.detect_transitional_period()


        transitional_period = self.data["global_analysis"]["transitional period"]
        cube_time = transitional_period["cube_time"]
        transition_north = transitional_period["transition_north"]
        transition_south = transitional_period["transition_south"]
        mean_transition = transitional_period["mean_transition"]
        plt.figure(1, figsize=[15,5])
        plt.title("Transition Wavelength (µm) vs time")
        plt.xlabel("Time (years)")
        
        # More robust tick handling
        min_year = int(np.min(cube_time))
        max_year = int(np.max(cube_time)) + 1
        
        # Set major ticks at yearly intervals
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
        
        # Disable minor ticks
        plt.gca().xaxis.set_minor_locator(plt.NullLocator())
        
        # Set the x-axis limits explicitly
        plt.xlim(min_year, max_year)
        ticks = range(min_year, max_year + 1, 1)
        plt.xticks(ticks)
        

        plt.ylabel("Transition Wavelength (µm)")
        plt.plot(cube_time, transition_north, label="Northern Transition")
        plt.plot(cube_time, transition_south, label="Southern Transition")
        plt.plot(cube_time, mean_transition, label="Mean Transition")
        plt.legend()
        fig_path = join_strings(self.fig_save_dir, self.fig_paths["transitional_period"])
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        plt.savefig(join_strings(fig_path,"all.png"), dpi = 300)
        # plt.show()
        plt.close()    
    def look_at_odd_data(self, range = None, non_fitted = False,plot = True):
        mu = np.linspace(0,1,100)
        fit_obj = fit_data()
        for index, (cube_name, cube_data) in enumerate(self.data.items()):
            leng = 0
            for ind, (wave_band, wave_data) in enumerate(cube_data.items()):
                if "µm_" not in wave_band:
                    continue
                if index in self.unusable_bands:
                    continue
                leng +=1
                try:
                    n = wave_data["north_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u1"] + wave_data["north_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u2"]
                    if range is not None and (n > abs(range) or n < -1*abs(range)):
                        plt.title("North Side" + " " + cube_name + " " + wave_band)
                        plt.plot(self.emission_to_normalized(wave_data["north_side"]["emission_angles"]), wave_data["north_side"]["brightness_values"], label = "North Side actual")
                        fitted_values = fit_obj.quadratic_limb_darkening(mu,*list(wave_data["north_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"].values()))
                        plt.plot(mu, fitted_values, label = "North Side found")
                        plt.legend()
                        plt.show()
                except:
                    if non_fitted:
                        plt.title("North Side" + " " + cube_name + " " + wave_band)
                        plt.plot(self.emission_to_normalized(wave_data["north_side"]["emission_angles"]), wave_data["north_side"]["brightness_values"], label = "North Side actual")
                        plt.legend()
                        plt.show()
                    continue
                try:
                    n = wave_data["south_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u1"] + wave_data["south_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u2"]
                    if range is not None and (n > abs(range) or n < -1*abs(range)):
                        plt.title("South Side" + " " + cube_name + " " + wave_band)
                        plt.plot(self.emission_to_normalized(wave_data["south_side"]["emission_angles"]), wave_data["south_side"]["brightness_values"], label = "South Side actual")
                        fitted_values = fit_obj.quadratic_limb_darkening(mu,*list(wave_data["south_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"].values()))
                        plt.plot(mu, fitted_values, label = "South Side found")
                        plt.legend()
                        plt.show()
                except:
                    if non_fitted:
                        plt.title("South Side" + " " + cube_name + " " + wave_band)
                        plt.plot(self.emission_to_normalized(wave_data["south_side"]["emission_angles"]), wave_data["south_side"]["brightness_values"], label = "South Side actual")
                        plt.legend()
                        plt.show()
                    continue

    def three_d_plot(self):
        fig = plt.figure()
        time = []
        wave = []
        north = []
        south = []
        for index, (cube_name, cube_data) in enumerate(self.data.items()):
            leng = 0
            if not self.check_if_cube(cube_name):
                continue
            for ind, (wave_band, wave_data) in enumerate(cube_data.items()):
                if "µm_" not in wave_band:
                    continue
                if index in self.unusable_bands:
                    continue
                leng +=1
                try:
                    n = wave_data["north_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u1"] + wave_data["north_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u2"]
                    if n > 10 or n < -10:
                        raise ValueError("Value too high")
                    north.append(n)
                except:
                    north.append(np.nan)
                try:
                    s = wave_data["south_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u1"] + wave_data["south_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u2"]
                    if s > 10 or s < -10:
                        raise ValueError("Value too high")
                    south.append(s)
                except:
                    south.append(np.nan)
                wave_length = wave_band.split("_")[0][0:-2]
                wave.append(float(wave_length))
            time.extend([self.get_time(cube_data["meta"]["cube_vis"]["time"])]*leng)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        north = np.array(north)
        ax.plot_trisurf(time, wave, north)
        ax.set_ylabel('Wavelength')
        ax.set_xlabel('Time')
        ax.set_zlabel('Value')
        ax.set_title('3D Tri-Surface Plot')
        plt.show()
        
    def trust_map(self, wave_length_cube, ground_array):
        size = np.mean(ground_array.shape)
        scalar = np.round(size/10).astype(int)
        ground_array = cv2.dilate(ground_array.astype(np.uint8),
                        np.ones((3, 3), np.uint8), iterations=scalar)
        ground = np.where(ground_array, np.nan, wave_length_cube)
        
        efficient_trimmed_array = ground.copy().flatten()[~np.isnan(ground.copy().flatten())]
        lower_bound = np.percentile(efficient_trimmed_array, 20)
        upper_bound = np.percentile(efficient_trimmed_array, 80)
        efficient_trimmed_array = efficient_trimmed_array[(efficient_trimmed_array > lower_bound) & (efficient_trimmed_array < upper_bound)]

        std = np.std(efficient_trimmed_array)
        mean = np.mean(wave_length_cube)
        # if abs(std/mean) > 1:
        #     nanm = np.nanmean(np.abs(ground))*3
        #     shows = np.where(np.isnan(ground), wave_length_cube+nanm, ground)
        #     plt.title(1 - abs(std/mean))
        #     plt.imshow(shows, cmap = "gray")
        #     plt.show()
        #     plt.close()
        return abs(std/mean)
    def trust_mapping(self):
        if "global_analysis" not in self.data:
            self.data["global_analysis"] = {}
        if "trust_map" in self.data["global_analysis"]:
            print("trust map already exists")
            return
        trust = {}
        trust_with_wave = [[] for i in range(352)]
        wavelengths = None
        
        cube_times = []
        cube_avgs = []
        cube_avgs_processed = []
        for index, (cube_name, cube_data) in enumerate(self.data.items()):
            if not self.check_if_cube(cube_name):
                continue
            if wavelengths == None:
                wavelengths = list(cube_data["meta"]["cube_vis"]["w"])
                wavelengths.extend(list(cube_data["meta"]["cube_ir"]["w"]))
            bands = []
            
            for ind, wave in enumerate(cube_data["meta"]["cube_vis"]["bands"]):
                ground = cube_data["meta"]["cube_vis"]["ground"]
                wave_trust = self.trust_map(wave, ground)
                bands.append(wave_trust)
                trust_with_wave[ind].append(wave_trust)
                
            for ind, wave in enumerate(cube_data["meta"]["cube_ir"]["bands"]):
                ground = cube_data["meta"]["cube_ir"]["ground"]
                wave_trust = self.trust_map(wave, ground)
                bands.append(wave_trust)
                trust_with_wave[ind+96].append(wave_trust)
            trust[cube_name] = bands
            cube_times.append(self.get_time(cube_data["meta"]["cube_vis"]["time"]))
            cube_avgs.append(np.mean(bands))
            cube_avgs_processed.append(np.mean([w for ind,w in enumerate(bands) if ind not in self.unusable_bands]))
            print("average trust for", cube_name, "is", np.mean(bands))
        trust_with_wave = [np.mean(trust) for trust in trust_with_wave]
        
        filtered_wave = [w for ind,w in enumerate(wavelengths) if ind not in self.unusable_bands]
        filtered_trust_with_wave = [w for ind,w in enumerate(trust_with_wave) if ind not in self.unusable_bands]

        # Set up publication-ready style
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'mathtext.fontset': 'stix',
            'font.size': 7,
            'axes.labelsize': 8,
            'axes.titlesize': 9,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 6,
            'figure.titlesize': 9,
            'axes.grid': True,
            'axes.grid.which': 'major',
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'axes.linewidth': 0.5,
            'axes.edgecolor': 'black',
            'lines.linewidth': 0.75,
            'figure.dpi': 300,
            'savefig.dpi': 600,
            'figure.figsize': (4.5, 2.6),
            'figure.constrained_layout.use': True
        })

        if not os.path.exists(join_strings(self.fig_save_dir, self.fig_paths["trust_map"])):
            os.makedirs(join_strings(self.fig_save_dir, self.fig_paths["trust_map"]))
            
        fig = plt.figure()
        ax = plt.gca()
        
        # Plot with improved styling
        ax.plot(wavelengths, [1/x for x in trust_with_wave], color='#3288BD', label='All wavelengths', linewidth=0.75)
        ax.plot(filtered_wave, [1/x for x in filtered_trust_with_wave], color='#D53E4F', label='Bands used for processing', linewidth=0.75)
        
        # Customize axes
        ax.set_xlabel('Wavelength (µm)')
        ax.set_ylabel('Signal to Noise ($\mu$/$\sigma$)')
        
        # Set x-axis ticks
        ax.set_xlim(0, 5.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
        
        # Set y-axis to start at 0
        ymin = 0
        ymax = max([1/x for x in trust_with_wave])
        # Round ymax up to nearest multiple of 100 for cleaner ticks
        ymax = np.ceil(ymax / 100) * 100
        ax.set_ylim(ymin, ymax)
        
        # Set major ticks every 100, minor every 20
        ax.yaxis.set_major_locator(plt.MultipleLocator(100))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(20))
        
        # Add grid with reduced density
        ax.grid(True, which='major', alpha=0.3, linewidth=0.5)
        ax.grid(True, which='minor', alpha=0.15, linewidth=0.5)
        
        # Customize legend
        ax.legend(frameon=True, fancybox=False, edgecolor='black', 
                 bbox_to_anchor=(1.0, 1.0), loc='upper right')
        
        plt.tight_layout()
        
        plt.savefig(join_strings(self.fig_save_dir, self.fig_paths["trust_map"],"trust_vs_wavelength.png"), 
                   dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.close()

        cube_times, cube_avgs, cube_avgs_processed = zip(*sorted(zip(cube_times, cube_avgs, cube_avgs_processed)))
        
        # Create time-based plot with same styling
        fig = plt.figure()
        ax = plt.gca()
        
        ax.plot(cube_times, [1/x for x in cube_avgs], color='#3288BD', label='All wavelengths', linewidth=0.75)
        ax.plot(cube_times, [1/x for x in cube_avgs_processed], color='#D53E4F', label='Bands used for processing', linewidth=0.75)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('Signal to Noise ($\mu$/$\sigma$)')
        
        # Set x-axis ticks
        min_year = int(np.min(cube_times))
        max_year = int(np.max(cube_times)) + 1
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.25))
        
        # Set y-axis to start at 0 with improved tick spacing
        ymin = 0
        ymax = max([1/x for x in cube_avgs_processed])
        # Round ymax up to nearest multiple of 100 for cleaner ticks
        ymax = np.ceil(ymax / 100) * 100
        ax.set_ylim(ymin, ymax)
        
        # Set major ticks every 100, minor every 20
        ax.yaxis.set_major_locator(plt.MultipleLocator(100))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(20))
        
        # Add grid with reduced density
        ax.grid(True, which='major', alpha=0.3, linewidth=0.5)
        ax.grid(True, which='minor', alpha=0.15, linewidth=0.5)
        
        # Customize legend
        ax.legend(frameon=True, fancybox=False, edgecolor='black',
                 bbox_to_anchor=(1.0, 1.0), loc='upper right')
        
        plt.tight_layout()
        
        plt.savefig(join_strings(self.fig_save_dir, self.fig_paths["trust_map"],"trust_vs_time.png"),
                   dpi=600, bbox_inches='tight', pad_inches=0.01)
        plt.close()

        self.data["global_analysis"]["trust_map"] = trust
        self.save(False)

    def stacked_u_vs_wave(self, cube_list=None):
        """
        Creates a stacked plot of u_vs_wavelength for multiple cubes.
        
        Parameters
        ----------
        cube_list : list, optional
            List of cube names to plot. If None, uses default cubes.
            
        Returns
        -------
        None
            Saves the figure to the holistic_stats subfolder.
        """
        if cube_list is None:
            # Default cubes that show interesting features or represent different periods
            cube_list = ["C1477456872_1", "C1559103132_1", "C1814906506_1"]
        
        # Set up publication-ready style
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'Times New Roman',
            'mathtext.fontset': 'stix',
            'font.size': 12,  # Increased from 8
            'axes.labelsize': 14,  # Increased from 9
            'axes.titlesize': 14,  # Increased from 10
            'xtick.labelsize': 12,  # Increased from 8
            'ytick.labelsize': 12,  # Increased from 8
            'legend.fontsize': 12,  # Increased from 8
            'figure.titlesize': 14,  # Increased from 10
            'axes.linewidth': 1.0,  # Increased from 0.5
            'axes.edgecolor': 'black',
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
        })
        
        # Create figure with subplots stacked vertically
        fig, axs = plt.subplots(len(cube_list), 1, figsize=(12, 4.7*len(cube_list)), sharex=False)
        if len(cube_list) == 1:
            axs = [axs]  # Make it iterable for single cube case
            
        # Add title to figure
        fig.suptitle("Seasonal Change in Limb Profiles vs. Wavelength", 
                    fontsize=16,  # Increased from 14
                    y=0.94)  # Lowered from 0.98
        
        for idx, (cube_name, ax) in enumerate(zip(cube_list, axs)):
            if cube_name not in self.data:
                print(f"Warning: Cube {cube_name} not found in data")
                continue
                
            cube_data = self.data[cube_name]
            shift = 0
            
            # Collect all wavelengths and values
            wavelengths = []
            north_values = []
            south_values = []
            
            # Process data for this cube
            for plot_index, (wave_band, wave_data) in enumerate(cube_data.items()):
                if "µm_" not in wave_band:
                    shift += 1
                    continue
                if plot_index in self.unusable_bands or (plot_index - shift + 1 > 96 and plot_index - shift + 1 < 108):
                    continue
                
                wavelength = float(wave_band.split("_")[0].replace("µm", ""))
                
                # Get the north and south data
                try:
                    nu1 = wave_data["north_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u1"]
                    nu2 = wave_data["north_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u2"]
                    north_val = nu1 + nu2
                except:
                    north_val = np.nan
                try:
                    su1 = wave_data["south_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u1"]
                    su2 = wave_data["south_side"]["fit"]["quadratic"]["optimal_fit"]["fit_params"]["u2"]
                    south_val = su1 + su2
                except:
                    south_val = np.nan
                
                wavelengths.append(wavelength)
                north_values.append(north_val)
                south_values.append(south_val)
            
            # Convert to numpy arrays for easier manipulation
            wavelengths = np.array(wavelengths)
            north_values = np.array(north_values)
            south_values = np.array(south_values)
            
            # Sort all arrays by wavelength
            sort_idx = np.argsort(wavelengths)
            wavelengths = wavelengths[sort_idx]
            north_values = north_values[sort_idx]
            south_values = south_values[sort_idx]
            
            # Find gaps in wavelength data
            gaps = np.where(np.diff(wavelengths) > 0.1)[0]
            
            # Plot data segments with connecting dashed lines
            start_idx = 0
            for gap_idx in gaps:
                # Plot solid lines for the current segment
                ax.plot(wavelengths[start_idx:gap_idx+1], north_values[start_idx:gap_idx+1], 
                       color=(1, 0.78, 0.33), linewidth=2)  # Increased from 1
                ax.plot(wavelengths[start_idx:gap_idx+1], south_values[start_idx:gap_idx+1], 
                       color=(0.23, 0.54, 0.54), linewidth=2)  # Increased from 1
                
                # Plot dashed lines connecting to next segment
                if gap_idx + 1 < len(wavelengths):
                    ax.plot([wavelengths[gap_idx], wavelengths[gap_idx+1]], 
                           [north_values[gap_idx], north_values[gap_idx+1]], 
                           color=(1, 0.78, 0.33), linestyle='--', linewidth=1.5, alpha=0.5)  # Increased from 0.75
                    ax.plot([wavelengths[gap_idx], wavelengths[gap_idx+1]], 
                           [south_values[gap_idx], south_values[gap_idx+1]], 
                           color=(0.23, 0.54, 0.54), linestyle='--', linewidth=1.5, alpha=0.5)  # Increased from 0.75
                
                start_idx = gap_idx + 1
            
            # Plot the final segment
            if start_idx < len(wavelengths):
                ax.plot(wavelengths[start_idx:], north_values[start_idx:], 
                       color=(1, 0.78, 0.33), linewidth=2)  # Increased from 1
                ax.plot(wavelengths[start_idx:], south_values[start_idx:], 
                       color=(0.23, 0.54, 0.54), linewidth=2)  # Increased from 1
            
            # Customize this subplot
            ax.set_xlim(0.25, 3.25)
            ax.set_ylim(-2.5, 0.75)  # Set consistent y-axis limits
            ax.set_ylabel("µ1 + µ2", fontsize=14)  # Increased from 9
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=1.5)  # Increased from 0.75
            
            # Set up grid
            ax.grid(True, which='major', alpha=0.15, linestyle='-', linewidth=0.5)
            ax.grid(True, which='minor', alpha=0.07, linestyle='-', linewidth=0.5)
            
            # Configure ticks for all plots
            ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
            ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
            ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
            ax.set_xlabel("Wavelength (µm)", fontsize=14)  # Added to all plots
            
            # Add cube name and date to top right
            cube_time = self.get_time(cube_data["meta"]["cube_vis"]["time"])
            ax.text(0.98, 0.98, f"{cube_name} ({cube_time:.2f})", 
                   transform=ax.transAxes, fontsize=12,  # Increased from 9
                   verticalalignment='top', horizontalalignment='right')
            
            # Add limb darkening/brightening text
            ax.text(0.5, 0.95, "Limb Darkening", transform=ax.transAxes,
                   ha='center', va='top', fontsize=12)  # Increased from 9
            ax.text(0.5, 0.05, "Limb Brightening", transform=ax.transAxes,
                   ha='center', va='top', fontsize=12)  # Increased from 9
        
        # Create legend at the bottom of figure, centered to plot area
        lines = [plt.Line2D([], [], color=(1, 0.78, 0.33), linewidth=2, label="North | µ1 + µ2"),
                plt.Line2D([], [], color=(0.23, 0.54, 0.54), linewidth=2, label="South | µ1 + µ2")]
        # Calculate legend position based on plot area
        legend = fig.legend(handles=lines, fontsize=12, frameon=True, fancybox=False, edgecolor='black',
                          bbox_to_anchor=(0.5, 0.02), loc='lower center', ncol=2,
                          bbox_transform=fig.transFigure)  # Added bbox_transform
        
        # Adjust layout with space at bottom for legend and reduced top space
        plt.subplots_adjust(hspace=0.25, left=0.1, right=0.95, top=0.92, bottom=0.09)  # Adjusted top to 0.93 for title
        
        # Create holistic_stats directory if it doesn't exist
        save_dir = join_strings(self.fig_save_dir, "holistic_stats")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Save figure
        cube_names = "_".join(cube_list)
        plt.savefig(join_strings(save_dir, f"stacked_u_vs_wave_{cube_names}.png"), 
                   dpi=600, bbox_inches='tight', pad_inches=0.02)
        plt.close()

