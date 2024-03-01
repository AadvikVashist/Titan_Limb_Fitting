from settings.get_settings import join_strings, check_if_exists_or_write, SETTINGS, get_cumulative_filename

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import datetime

class timeline_figure:
    def __init__(self, devEnvironment: bool = True):
        if devEnvironment == True:
            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["dev_figures_sub_path"],  SETTINGS["paths"]["figure_subpath"]["image_overlay"])
        else:
            self.dps_dir = join_strings(
                SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["prod_figures_sub_path"],  "DPS_55")

            self.save_dir = join_strings(
                SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["prod_figures_sub_path"],  SETTINGS["paths"]["figure_subpath"]["image_overlay"])
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

    def filter_to_bands_and_cubes(self,cubes,bands):
        years = []
        data_points = {}
        for cube, cube_data in cubes.items():
            band_inds = [band_name for index, band_name in enumerate(list(cube_data.keys())) if "µm" in band_name and int(band_name.split("_")[1]) in bands]
            if len(band_inds) != len(bands):
                raise ValueError(f"Cube {cube} does not have all bands {bands} available")
            # band_wave = wave_band.split("_")[1] + "_" + wave_band.split("_")[0]
            waves = [band_ind.split("_")[0] for band_ind in band_inds]
            year = self.get_time(cube_data["meta"]["cube_vis"]["time"])
            for index, band_ind in enumerate(band_inds):
                north_slant = cube_data[band_ind]["north_side"]
                south_slant = cube_data[band_ind]["south_side"]
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

                if bands[index] not in data_points:
                    data_points[bands[index]] = [[nu1+nu2, su1+su2]]
                else:
                    data_points[bands[index]].append([nu1+nu2, su1+su2])
            years.append(year)
        return data_points, years, waves
    def gen_figures(self):
        bands_to_use=[46, 118, 200]

        n_seasons = ['Spring', 'Summer', 'Fall']
        s_seasons = ['Fall', 'Winter', 'Spring']

        northern_starts = [2003, 2009.5, 2017.25]

        y_ticks_min = -4
        y_ticks_max = 2
        y_ticks_minor_step = 0.25
        y_ticks_major_step = 1


        y_major_ticks = np.arange(y_ticks_min, y_ticks_max + y_ticks_major_step, y_ticks_major_step)
        y_minor_ticks = np.arange(y_ticks_min, y_ticks_max + y_ticks_minor_step, y_ticks_minor_step)

        x_ticks_min = 2004
        x_ticks_max = 2018
        x_ticks_minor_step = 1
        x_ticks_major_step = 4

        x_major_ticks = np.arange(x_ticks_min, x_ticks_max + x_ticks_major_step, x_ticks_major_step)
        if x_ticks_max not in x_major_ticks:
            x_major_ticks = np.append(x_major_ticks, x_ticks_max)
        x_minor_ticks = np.arange(x_ticks_min, x_ticks_max + x_ticks_minor_step, x_ticks_minor_step)


        all_data = self.get_data()
        usage_data,years,waves = self.filter_to_bands_and_cubes(all_data, bands_to_use)

        # Create the figure and axis again with the specified constraints
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.axhline(y=0, color='k', linestyle='--', alpha=1, linewidth=2)


        # Plot the data points with T-shaped error bars
        for index, wave in enumerate(waves):
            data_points = np.array(usage_data[bands_to_use[index]])
            ax.plot(years, data_points[:,0], label=f'{wave} N') #'r', 
            ax.plot(years, data_points[:,1], label=f'{wave} S') #'b', 
        # ax.errorbar(years, data_points, yerr=error, fmt='ro', ecolor='black', elinewidth=1, capsize=5, capthick=2)
        # Set the labels for the axes
        ax.set_xlabel('Year')
        ax.set_ylabel('µ1 + µ2')

        # Set x-axis major ticks every 5 years and minor ticks every year
        ax.set_xticks(x_major_ticks)
        ax.set_xticks(x_minor_ticks, minor=True)

        ax.set_yticks(y_major_ticks, minor=False)
        ax.set_yticks(y_minor_ticks, minor=True)

        # Draw vertical lines only for the specified start dates of each season in the Northern hemisphere
        for start_year in northern_starts:
            ax.axvline(x=start_year, color='k', linestyle='--', alpha=1, linewidth=3)
        # Add season labels for the Northern hemisphere
        for i, start_year in enumerate(northern_starts):
            if i < len(northern_starts) - 1:
                mid_point = (start_year + northern_starts[i + 1]) / 2
            else:
                # For the last northern season, estimate a reasonable end point
                mid_point = (start_year + x_ticks_max) / 2
            
            label = f'N {n_seasons[i]}'
            ax.text(mid_point, y_ticks_max-y_ticks_minor_step, label, ha='center', va='center', backgroundcolor='white', weight='bold', fontsize=12)
            label = f'S {s_seasons[i]}'
            ax.text(mid_point, y_ticks_min+y_ticks_minor_step, label, ha='center', va='center', backgroundcolor='white', weight='bold', fontsize=12)

        # Ensure y-values are constrained between 0 and 10
        ax.set_xlim(x_ticks_min, x_ticks_max)
        ax.set_ylim(y_ticks_min, y_ticks_max)
        ax.legend()
        plt.title('Timeline with Data Points, T-shaped Error Bars, and Seasonal Markers')
        plt.grid(True, which='major', axis='both')
        plt.grid(True, which='minor', axis='x')
        plt.tight_layout()
        plt.show()
