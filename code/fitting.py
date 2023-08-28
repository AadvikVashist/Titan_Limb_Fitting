from polar_profile import analyze_complete_dataset
from sort_and_filter import sort_and_filter
from get_settings import join_strings, check_if_exists_or_write, SETTINGS
import re
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter


class fit_data:
    def __init__(self, fit_type: str = "default"):
        self.save_dir = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["fitted_sub_path"])
        self.data_dir = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["sorted_sub_path"])

        if fit_type == "linear" or fit_type == "lin":
            self.limb_darkening_function = self.linear_limb_darkening
            self.limb_darkening_function_name = "lin"
        elif fit_type == "quadratic" or fit_type == "quad":
            self.limb_darkening_function = self.quadratic_limb_darkening
            self.limb_darkening_function_name = "quad"
        elif fit_type == "sqrt" or fit_type == "square_root":
            self.limb_darkening_function = self.square_root_limb_darkening
            self.limb_darkening_function_name = "sqrt"
        else:
            self.limb_darkening_function = self.quadratic_limb_darkening
            self.limb_darkening_function_name = "quad"

    def emission_to_normalized(self, emission_angle):
        return np.cos(np.deg2rad(emission_angle))

    def linear_limb_darkening(self, r, u):
        """
        Calculate intensity using the linear limb darkening law.

        Parameters:
            r (array): Normalized distances from the center of Titan.
            I_0 (float): Intensity at the center of Titan.
            u (float): Linear limb darkening coefficient.

        Returns:
            array: Intensity values corresponding to the distances in r.
        """
        return self.I_0 * (1 - u * (1 - r))

    def quadratic_limb_darkening(self, r, u_1, u_2):
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
        return self.I_0 * (1 - (u_1 * (1 - r)) - (u_2 * ((1 - r)**2)))

    def square_root_limb_darkening(self, r, u_1, u_2):
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
        return self.I_0 * (1 - u_1 * (1 - r) - u_2 * (1 - np.sqrt(r)))

    def get_filtered_data(self):
        all_data = {}
        if os.path.exists(join_strings(self.data_dir, SETTINGS["paths"]["cumulative_fitted_path"])):
            all_data = check_if_exists_or_write(join_strings(
                self.data_dir, SETTINGS["paths"]["cumulative_sorted_path"]), save=False, verbose=True)
        else:
            cubs = os.listdir(self.data_dir)
            cubs = [cub for cub in cubs if re.fullmatch(
                r'C.*_.*\.pkl', cub) is not None]
            for cub in cubs:
                cube_name = os.path.splitext(cub)[0]
                all_data[cube_name] = check_if_exists_or_write(
                    join_strings(self.data_dir, cub), save=False, verbose=True)
        return all_data

    def limb_darkening_laws(self, emission_angles, brightness_values):
        """
        Fit the brightness values to the limb darkening laws.
        returns: 
            (I_0, u_1, u_2) (tuple): Fitted parameters for the limb darkening laws.
                I_0 (float): Intensity at the center of Titan.
                u_1, u_2 (float): Linear and quadratic limb darkening coefficients. (u2 is 0 if linear limb darkening is used)
            pcov (ndarray): Covariance matrix of the fitted parameters.

        """
        distances = self.emission_to_normalized(emission_angles)

        I_0 = np.max(brightness_values)
        if self.limb_darkening_function_name == "lin":
            # param_bounds = ([0.8333 * I_0, -np.inf], [1.2 * I_0, np.inf])
            p0 = [-1]
            self.I_0 = I_0
            popt, pcov = curve_fit(self.limb_darkening_function, distances, brightness_values,
                                   full_output=False, p0=p0, method="trf")
            plt.plot(distances, brightness_values)
            plt.plot(distances, self.limb_darkening_function(
                distances, popt[0]))
            plt.show()
            return {"I_0":  self.I_0, "u": popt[0]}, pcov, distances
        elif self.limb_darkening_function_name == "quad":
            self.I_0 = I_0

            param_bounds = ([0, -1], [2, 1])
            p0 = [1, -1]
            self.I_0 = I_0

            popt, pcov = curve_fit(self.limb_darkening_function, distances, brightness_values,
                                   full_output=False, bounds=param_bounds, p0=p0, method="trf")
            # plt.plot(distances, brightness_values)
            # plt.plot(distances, self.limb_darkening_function(
            #     distances, popt[0], popt[1]))

            # # Smooth data using moving average
            # plt.show()
            return {"I_0": self.I_0, "u1": popt[0], "u2": popt[1]}, pcov
        elif self.limb_darkening_function_name == "sqrt":
            return {"I_0": self.I_0, "u1": popt[0], "u2": popt[1]}, pcov

    def fit(self, data: dict):
        leng = len(data)
        for index, (wave_band, wave_data) in enumerate(data.items()):
            for slant, slant_data in wave_data.items():
                emission_angles = np.array(slant_data["emission_angles"])
                brightness_values = np.array(slant_data["brightness_values"])
                pchip = PchipInterpolator(emission_angles, brightness_values)
                interp_x = np.linspace(
                    np.min(emission_angles), np.max(emission_angles), 100)
                interp_y = pchip(interp_x)
                popt, pcov = self.limb_darkening_laws(
                    interp_x, interp_y)
                
                # fit_obj = fit_data()
                # fit_obj.I_0 = popt["I_0"]
                # plt.plot(fit_obj.emission_to_normalized(emission_angle=emission_angles), brightness_values)
                # plt.plot(fit_obj.emission_to_normalized(emission_angle=emission_angles), self.limb_darkening_function(
                #     fit_obj.emission_to_normalized(emission_angle=emission_angles), popt["u1"], popt["u2"]))
                # plt.show()
                data[wave_band][slant]["meta"]["processing"]["fitted"] = True
                data[wave_band][slant]["fit"] = {
                    "fit_params": popt, "covariance_matrix": pcov, "limb_darkening_function": self.limb_darkening_function_name}
                # base_fitted_y = [self.limb_darkening_function(1 - np.cos(np.deg2rad(xs)), popt["u1"], popt["u2"]) for xs in interp_x]

                # plt.plot(emission_angles, brightness_values, label="data")
                # plt.plot(interp_x, interp_y, label="chip")
                # plt.plot(interp_x, base_fitted_y, label="base_fit")
                # plt.legend()
                # plt.show()
                # data[wave_band][slant]["pixel_indices"], data[wave_band][slant]["pixel_distances"], data[wave_band][slant]["emission_angles"], data[wave_band][slant]["brightness_values"]
            print("Finished fitting", wave_band, "| Spent", np.around(time.time() - self.cube_start_time, 3), "| expected time left:",
                  np.around((time.time() - self.cube_start_time) * (index + 1) * (leng - index - 1), 2), end="\r")
        print()
        return data

    def fit_all(self):
        data = self.get_filtered_data()
        force_write = (SETTINGS["processing"]["clear_cache"]
                       or SETTINGS["processing"]["redo_fitting"])
        appended_data = False
        cube_count = len(data)
        self.start_time = time.time()

        for index, (cube_name, cube_data) in enumerate(data.items()):
            if os.path.exists(join_strings(self.save_dir, cube_name + ".pkl")) and not force_write:
                try:
                    data[cube_name] = check_if_exists_or_write(
                        join_strings(self.save_dir, cube_name + ".pkl"), save=False)
                    print("fitted data already exists. Skipping...")
                    continue
                except:
                    print("fitted data corrupted. Processing...")
            elif not force_write:
                appended_data = True
            self.cube_start_time = time.time()
            # only important line in this function
            data[cube_name] = self.fit(cube_data)

        # Calculate expected time left based on the average time per cube
            check_if_exists_or_write(join_strings(
                self.save_dir, cube_name + ".pkl"), data=data[cube_name], save=True, force_write=True)
            print("Total time for cube: ", np.around(time.time() - self.cube_start_time, 3), "seconds | Expected time left:",
                  np.around((time.time() - self.start_time) / (index + 1) * (cube_count - index - 1), 2), "seconds")

        if os.path.exists(join_strings(self.save_dir, SETTINGS["paths"]["cumulative_fitted_path"])) and (not force_write or appended_data):
            print("Since fitted data already exists, but new data has been appended ->")
            check_if_exists_or_write(join_strings(
                self.save_dir, SETTINGS["paths"]["cumulative_fitted_path"]), data=data, save=True, force_write=True, verbose=True)
        else:
            check_if_exists_or_write(join_strings(
                self.save_dir, SETTINGS["paths"]["cumulative_fitted_path"]), data=data, save=True, force_write=True, verbose=True)
