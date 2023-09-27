from .polar_profile import analyze_complete_dataset
from .sort_and_filter import sort_and_filter
from get_settings import join_strings, check_if_exists_or_write, SETTINGS, get_cumulative_filename
import re
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter
import pyvims
from sklearn.metrics import r2_score


class select_data:
    def __init__(self):
        self.save_dir = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["selected_sub_path"])
        self.data_dir = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["fitted_sub_path"])

    def emission_to_normalized(self, emission_angle):
        return np.cos(np.deg2rad(emission_angle))

    def linear_limb_darkening(self, r, I_0, u):
        """
        Calculate intensity using the linear limb darkening law.

        Parameters:
            r (array): Normalized distances from the center of Titan.
            I_0 (float): Intensity at the center of Titan.
            u (float): Linear limb darkening coefficient.

        Returns:
            array: Intensity values corresponding to the distances in r.
        """
        return I_0 * (1 - u * (1 - r))

    def quadratic_limb_darkening(self, r, I_0, u_1, u_2):
        """
        Calculate intensity using the quadratic limb darkening law.

        Parameters:
            r (array): Normalized distances from the center of Titan.
            I_0 (float): Intensity at the center of Titan.
            u_1 (float): Linear limb darkening coefficient.
            u_2 (float): Quadratic limb darkening coefficient.

        Returns:
            array: Intensity values corresponding to the distances in r.
        """
        return I_0 * (1 - (u_1 * (1 - r)) - (u_2 * ((1 - r)**2)))

    def square_root_limb_darkening(self, r, I_0, u_1, u_2):
        """
        Calculate intensity using the square root limb darkening law.

        Parameters:
            r (array): Normalized distances from the center of Titan.
            I_0 (float): Intensity at the center of Titan.
            u_1 (float): Linear limb darkening coefficient.
            u_2 (float): Quadratic limb darkening coefficient.

        Returns:
            array: Intensity values corresponding to the distances in r.
        """
        return I_0 * (1 - u_1 * (1 - r) - u_2 * (1 - np.sqrt(r)))

    def get_fitted_data(self):
        all_data = {}
        if os.path.exists(join_strings(self.data_dir, get_cumulative_filename("fitted_sub_path"))):
            all_data = check_if_exists_or_write(join_strings(
                self.data_dir, get_cumulative_filename("fitted_sub_path")), save=False, verbose=True)
        else:
            cubs = os.listdir(self.data_dir)
            cubs = [cub for cub in cubs if re.fullmatch(
                r'C.*_.*\.pkl', cub) is not None]
            for cub in cubs:
                cube_name = os.path.splitext(cub)[0]
                all_data[cube_name] = check_if_exists_or_write(
                    join_strings(self.data_dir, cub), save=False, verbose=True)
        return all_data

    # def limb_darkening_law(self, emission_angles, brightness_values):
        """
        Fit the brightness values to the limb darkening laws.
        returns: 
            (I_0, u_1, u_2) (tuple): Fitted parameters for the limb darkening laws.
                I_0 (float): Intensity at the center of Titan.
                u_1, u_2 (float): Linear and quadratic limb darkening coefficients. (u2 is 0 if linear limb darkening is used)
            pcov (ndarray): Covariance matrix of the fitted parameters.

        """
        # distances = self.emission_to_normalized(emission_angles)

        # if self.limb_darkening_function_name == "lin":
        #     # param_bounds = ([0.8333 * I_0, -np.inf], [1.2 * I_0, np.inf])
        #     p0 = [-1]
        #     popt, pcov = curve_fit(self.limb_darkening_function, distances, brightness_values,
        #                            full_output=False, p0=p0, sigma=100/brightness_values, absolute_sigma=False)
        #     plt.plot(distances, brightness_values)
        #     plt.plot(distances, self.limb_darkening_function(
        #         distances, popt[0]))
        #     plt.show()
        #     return {"I_0":  self.I_0, "u": popt[0]}, pcov, distances
        # elif self.limb_darkening_function_name == "quad":

        #     # param_bounds = ([0, -1], [2, 1])
        #     p0 = [1, 0.5, 0.5]
        #     param_bounds = [[0, -1, -1], [np.inf, 1, 1]]
        #     try:
        #         popt, pcov = curve_fit(self.limb_darkening_function, distances, brightness_values,
        #                            full_output=False, p0=p0, bounds=param_bounds)
        #         return {"I_0": popt[0], "u1": popt[1], "u2": popt[2]}, pcov
        #     except:

        #         return distances, "error occured"
        # elif self.limb_darkening_function_name == "sqrt":
        # return {"I_0": self.I_0, "u1": popt[0], "u2": popt[1]}, pcov
    def run_linear_limb_darkening(self, normalized_distances, brightness_values):
        # param_bounds = ([0.8333 * I_0, -np.inf], [1.2 * I_0, np.inf])
        p0 = [1, 0.5]
        try:
            popt, pcov = curve_fit(self.linear_limb_darkening, normalized_distances, brightness_values,
                                   full_output=False, p0=p0)
            return {"I_0": popt[0], "u": popt[1]}, pcov
        except Exception as e:
            return e, "error occured"

    def run_quadratic_limb_darkening(self, normalized_distances, brightness_values):
        p0 = [1, 0.5, 0.5]
        param_bounds = [[0, -1, -1], [np.inf, 1, 1]]
        try:
            popt, pcov = curve_fit(self.quadratic_limb_darkening, normalized_distances,
                                   brightness_values, full_output=False, p0=p0, bounds=param_bounds)
            return {"I_0": popt[0], "u1": popt[1], "u2": popt[2]}, pcov
        except Exception as e:
            return e, "error occured"

    def run_square_root_limb_darkening(self, normalized_distances, brightness_values):
        p0 = [1, 0.5, 0.5]
        try:
            popt, pcov = curve_fit(self.square_root_limb_darkening,
                                   normalized_distances, brightness_values, full_output=False, p0=p0)
            return {"I_0": popt[0], "u1": popt[1], "u2": popt[2]}, pcov
        except Exception as e:
            return e, "error occured"

    def use_inc_to_select_fits(self, wave_band, wave_data, rot_angle):
        x = 0
        if rot_angle <= 180:
            ret_value = {60: wave_data[60], 120: wave_data[120],
                         "north_side": wave_data[60], "south_side": wave_data[120]}
            ret_value["north_side"]["angle"] = 60
            ret_value["south_side"]["angle"] = 120
        elif rot_angle > 180:
            ret_value = {300: wave_data[300], 240: wave_data[240],
                         "north_side": wave_data[300], "south_side": wave_data[240]}
            ret_value["north_side"]["angle"] = 300
            ret_value["south_side"]["angle"] = 240

        return ret_value


    def update_cube(self, data: dict, cube_name: str = None):
        leng = len(data.keys()) - 1

        for index, (wave_band, wave_data) in enumerate(data.items()):
            bad = False
            if "µm_" not in wave_band:
                continue
            if index < 96:
                ret_values = self.use_inc_to_select_fits(
                    wave_band, wave_data, data["meta"]["lowest_inc_vis"])
            else:
                ret_values = self.use_inc_to_select_fits(
                    wave_band, wave_data, data["meta"]["lowest_inc_ir"])
            data[wave_band] = ret_values

            time_spent = np.around(time.time() - self.cube_start_time, 3)
            percentage_completed = (index + 1) / leng
            total_time_left = time_spent / percentage_completed - time_spent

            print("Finished adjusting", wave_band, "| Spent", time_spent, " so far | expected time left:",
                  np.around(total_time_left, 2), "| total time for cube :",  np.around(total_time_left + time_spent, 3), end="\r")
        print()
        return data

    def check_stats(self):
        data = self.get_fitted_data()
        count = 0
        total = 0
        for cube, cube_data in data.items():
            for wave_band, wave_data in cube_data.items():
                if "µm_" not in wave_band:
                    continue
                for slant, slant_data in wave_data.items():
                    total += 1
                    if slant_data["meta"]["processing"]["fitted"]:
                        count+=1
        print(count, total, count/total)
    def run_selection_on_all(self, fit_types: str = "all"):
        data = self.get_fitted_data()
        force_write = (SETTINGS["processing"]["clear_cache"]
                       or SETTINGS["processing"]["redo_selection"])
        appended_data = False
        cube_count = len(data)
        self.start_time = time.time()
        self.fit_types = fit_types
        for index, (cube_name, cube_data) in enumerate(data.items()):
            if os.path.exists(join_strings(self.save_dir, cube_name + ".pkl")) and not force_write:
                try:
                    data[cube_name] = check_if_exists_or_write(
                        join_strings(self.save_dir, cube_name + ".pkl"), save=False)
                    print("selected data already exists. Skipping...")
                    continue
                except:
                    print("selected data corrupted. Processing...")
            elif not force_write:
                appended_data = True
            self.cube_start_time = time.time()
            # only important line in this function
            data[cube_name] = self.update_cube(cube_data, cube_name)

        # Calculate expected time left based on the average time per cube
            check_if_exists_or_write(join_strings(
                self.save_dir, cube_name + ".pkl"), data=data[cube_name], save=True, force_write=True)

            time_spent = np.around(time.time() - self.cube_start_time, 3)
            percentage_completed = (index + 1) / cube_count
            total_time_left = time_spent / percentage_completed - time_spent
            print("Cube", index + 1, "of", cube_count, "| Total time for cube:", time_spent, "seconds | Total Expected time left:",
                  np.around(total_time_left, 2), "seconds", "| Total time spent:", np.around(time.time() - self.start_time, 3), "seconds")
        if (os.path.exists(join_strings(self.save_dir, get_cumulative_filename("selected_sub_path"))) and appended_data):
            print("Fitted data already exists, but new data has been appended")
            check_if_exists_or_write(join_strings(
                self.save_dir, get_cumulative_filename("selected_sub_path")), data=data, save=True, force_write=True)
        elif force_write:
            check_if_exists_or_write(join_strings(
                self.save_dir, get_cumulative_filename("selected_sub_path")), data=data, save=True, force_write=True)
        else:
            print("Fitted not changed since last run. No changes to save...")
