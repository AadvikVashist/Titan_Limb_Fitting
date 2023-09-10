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
import pyvims
from sklearn.metrics import r2_score
def simple_moving_average(x, N):
    result = []
    window = []
    #check if array is increasing or decreasing
    if np.mean(x[0:int(len(x)/2)]) > np.mean(x[int(len(x)/2):-1]):
        x = x[::-1]
        back = True
    else:
        back = False
    for i in range(len(x)):
        window.append(x[i])
        if len(window) >= N:
            window.pop(0)
        result.append(np.mean(window))
    if back:
        result = result[::-1]
    return np.array(result)
class fit_data:
    def __init__(self):
        self.save_dir = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["fitted_sub_path"])
        self.data_dir = join_strings(
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["sorted_sub_path"])



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

    def get_filtered_data(self):
        all_data = {}
        if os.path.exists(join_strings(self.data_dir, SETTINGS["paths"]["cumulative_sorted_path"])):
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
            popt, pcov = curve_fit(self.quadratic_limb_darkening, normalized_distances, brightness_values,full_output=False, p0=p0, bounds=param_bounds)
            return {"I_0": popt[0], "u1": popt[1], "u2": popt[2]}, pcov
        except Exception as e:
            return e, "error occured"
    def run_square_root_limb_darkening(self, normalized_distances, brightness_values):
        p0 = [1, 0.5, 0.5]
        try:
            popt,pcov = curve_fit(self.square_root_limb_darkening, normalized_distances, brightness_values,full_output=False, p0=p0)
            return {"I_0": popt[0], "u1": popt[1], "u2": popt[2]}, pcov
        except Exception as e:
            return e, "error occured"
        
        
    def limb_darkening_laws(self, emission_angles, brightness_values, fit_types):
        #get the interpolated data by normalizing the emission angles, and sorting the data
        emission_angles_to_normalized = self.emission_to_normalized(emission_angles)
        pairs = list(zip(emission_angles_to_normalized, np.array(brightness_values)))

        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        #the actual interpolation     
        pchip = PchipInterpolator(*zip(*sorted_pairs))
        range = np.max(emission_angles_to_normalized) - np.min(emission_angles_to_normalized)
        interp_x = list(np.linspace(np.min(emission_angles_to_normalized), np.min(emission_angles_to_normalized) + 0.15 * range, 50)); interp_x.extend(np.linspace(np.min(emission_angles_to_normalized) + 0.15 * range, np.max(emission_angles_to_normalized) - 0.15 * range, 100)); interp_x.extend(np.linspace(np.max(emission_angles_to_normalized) - 0.15 * range, np.max(emission_angles_to_normalized), 50))
        interp_y = pchip(interp_x)
        
        #get the smoothed data
        sigma = 20
        window = 20
        gauss = gaussian_filter(interp_y, sigma=sigma)
        sma = simple_moving_average(interp_y, N = window)
        

        if fit_types == "all":
            fit_types = ["lin", "quad", "sqrt"]
        #fit the data
        elif type(fit_types) == str:
            fit_types = [fit_types]


        return_value = {}
        for fit_type in fit_types:
            if fit_type == "lin":
                fit_type = "linear"
                run_func = self.run_linear_limb_darkening
                plot_func = self.linear_limb_darkening
            elif fit_type == "quad":
                fit_type = "quadratic"
                run_func = self.run_quadratic_limb_darkening
                plot_func = self.quadratic_limb_darkening
            elif fit_type == "sqrt":
                fit_type = "square root"
                run_func = self.run_square_root_limb_darkening
                plot_func = self.square_root_limb_darkening
            standard_popt, standard_pcov = run_func(interp_x, interp_y)
            gauss_popt, gauss_pcov = run_func(interp_x, gauss)
            sma_popt, sma_pcov = run_func(interp_x, sma)
            if type(standard_pcov) == str:
                fit_fit = None
                fit_r2 = 0
            else:
                fit_fit = plot_func(emission_angles_to_normalized, *standard_popt.values())
                fit_r2 = r2_score(brightness_values, fit_fit)
            
            if type(gauss_pcov) == str:
                gauss_fit = None
                gauss_r2 = 0
            else:
                gauss_fit = plot_func(emission_angles_to_normalized, *gauss_popt.values())
                gauss_r2 = r2_score(brightness_values, gauss_fit)
            
            if type(sma_pcov) == str:
                sma_fit = None
                sma_r2 = 0
            else:
                sma_fit = plot_func(emission_angles_to_normalized, *sma_popt.values())
                sma_r2 = r2_score(brightness_values, sma_fit)

            
            
            if gauss_r2 > fit_r2 and gauss_r2 > sma_r2:
                optimal_fit = {"fit_params": gauss_popt, "covariance_matrix": gauss_pcov, "r2": gauss_r2, "type": "gaussian", "sigma": sigma}
            elif sma_r2 > fit_r2 and sma_r2 > gauss_r2:
                optimal_fit = {"fit_params": sma_popt, "covariance_matrix": sma_pcov, "r2": sma_r2, "type": "moving average", "window": window}
            else:
                optimal_fit = {"fit_params": standard_popt, "covariance_matrix": standard_pcov, "r2": fit_r2, "type": "standard"}
            if all([gauss_r2 == 0, sma_r2 == 0, fit_r2 == 0]):
                optimal_fit = []
            return_value[fit_type] = {
            "standard_fit" : {"fit_params": standard_popt, "covariance_matrix": standard_pcov, "r2": fit_r2},
            "gaussian_fit" : {"fit_params": gauss_popt, "covariance_matrix": gauss_pcov, "r2": gauss_r2, "sigma": sigma},
            "moving_average_fit" : {"fit_params": sma_popt, "covariance_matrix": sma_pcov, "r2": sma_r2, "window": window},
            "optimal_fit" : optimal_fit,
            }
        # # fit_fit = self.limb_darkening_function(emission_angles_to_normalized, standard_popt["I_0"], standard_popt["u1"], standard_popt["u2"])
        # gaus_fit = self.limb_darkening_function(emission_angles_to_normalized, gauss_popt["I_0"], gauss_popt["u1"], gauss_popt["u2"])
        # sma_fit = self.limb_darkening_function(emission_angles_to_normalized, sma_popt["I_0"], sma_popt["u1"], sma_popt["u2"])
        
        # # plt.plot(emission_angles_to_normalized, brightness_values, label="data")
        # # plt.plot(emission_angles_to_normalized, fit_fit, label="fit")
        # # plt.plot(emission_angles_to_normalized, gaus_fit, label="gaus")
        # # plt.plot(emission_angles_to_normalized, sma_fit, label="sma")
        # # plt.legend()
        # # plt.ylim(bottom = 0)
        # # plt.show()
        # gauss_r2 = r2_score(brightness_values, gaus_fit)
        # sma_r2 = r2_score(brightness_values, sma_fit)
        
        # if gauss_r2 > fit_r2 and gauss_r2 > sma_r2:
        #     optimal_popt, optimal_pcov = gauss_popt, gauss_pcov
        # elif sma_r2 > fit_r2 and sma_r2 > gauss_r2:
        #     optimal_popt, optimal_pcov = sma_popt, sma_pcov
        # else:
        #     optimal_popt, optimal_pcov = standard_popt, standard_pcov
        return return_value

    def fit(self, data: dict, cube_name: str = None):
        leng = len(data.keys())
        cube_vis = pyvims.VIMS(cube_name + "_vis.cub", join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube_name), channel="vis")
        cube_ir = pyvims.VIMS(cube_name + "_ir.cub",  join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube_name), channel="ir")
        for index, (wave_band, wave_data) in enumerate(data.items()):
            bad = False
            for slant, slant_data in wave_data.items():
                emission_angles = np.array(slant_data["emission_angles"])
                brightness_values = np.array(slant_data["brightness_values"])
                ret_values = self.limb_darkening_laws(emission_angles, brightness_values, self.fit_types)
                if all([len(ret_value["optimal_fit"]) == 0 for ret_value in ret_values.values()]):
                    data[wave_band][slant]["meta"]["processing"]["fitted"] = False
                else:
                    data[wave_band][slant]["meta"]["processing"]["fitted"] = True
                data[wave_band][slant]["fit"] = ret_values

                
                
                # fig, axs = plt.subplots(1,4, figsize= (15,5))
                # plt.title(cube_name + "  " + wave_band + " " + str(slant))
                # # axs[0].plot(interp_x, interp_y)
                # axs[0].plot(self.emission_to_normalized(emission_angle=emission_angles), brightness_values, label="data")
                # axs[0].plot(self.emission_to_normalized(emission_angle=emission_angles), self.limb_darkening_function(
                #     self.emission_to_normalized(emission_angle=emission_angles), popt["I_0"],  popt["u1"], popt["u2"]), label="fit")
                
                
                # axs[1].plot(self.emission_to_normalized(emission_angle=emission_angles), brightness_values, label="data")
                # axs[1].plot(self.emission_to_normalized(emission_angle=interp_x), gaus, label="gaus")

                # axs[1].plot(self.emission_to_normalized(emission_angle=interp_x), self.limb_darkening_function(
                #     self.emission_to_normalized(emission_angle=interp_x), popt["I_0"],  popt["u1"], popt["u2"]), label="fit_gaus")
                
                # axs[2].plot(self.emission_to_normalized(emission_angle=emission_angles), brightness_values, label="data")
                # axs[2].plot(self.emission_to_normalized(emission_angle=interp_x), sma, label="sma")

                # axs[2].plot(self.emission_to_normalized(emission_angle=interp_x), self.limb_darkening_function(
                #     self.emission_to_normalized(emission_angle=interp_x), popt["I_0"],  popt["u1"], popt["u2"]), label="fit_sma")
                
                
                # if index < 96:
                #     axs[3].imshow(cube_vis[index+1])
                # else:
                #     axs[3].imshow(cube_ir[index+1])
                # axs[0].legend()
                # axs[1].legend()
                # axs[2].legend()
                # axs[0].set_ylim(bottom = 0)
                # axs[1].set_ylim(bottom = 0)
                # axs[2].set_ylim(bottom = 0)
                # plt.show()
                
                # if type(standard_pcov) == str and standard_pcov == "error occured":
                #     # distances = popt
                #     # fig, axs = plt.subplots(1,2)
                #     # plt.title(cube_name + "  " + wave_band + " " + str(slant))
                #     # axs[0].plot(distances, interp_y)
                #     # if index < 96:
                #     #     axs[1].imshow(cube_vis[index+1])
                #     # else:
                #     #     axs[1].imshow(cube_ir[index+1])
                #     # plt.plot(distances, self.limb_darkening_function(
                #     #     distances, popt[0], popt[1]))
                #     # # Smooth data using moving average
                    
                #     # plt.show()
                #     data[wave_band][slant]["meta"]["processing"]["fitted"] = None
                #     data[wave_band][slant]["fit"] = {}
                #     bad = False 

                # else:
                    # x = 0
                    # standard_popt, standard_pcov = self.limb_darkening_law(interp_x, interp_y)

                    # popt_gaus, pcov_gaus = self.limb_darkening_law(interp_x, gaus)
                    # popt_sma, pcov_sma = self.limb_darkening_law(interp_x, sma)
                    
                    # gaus = gaussian_filter(interp_y, sigma=20)
                    # window_size = 20
                    # sma = simple_moving_average(interp_y, window_size)
                    # emission_angles_to_normalized = self.emission_to_normalized(emission_angles)
                    # fit_fit = self.limb_darkening_function(emission_angles_to_normalized, standard_popt["I_0"], standard_popt["u1"], standard_popt["u2"])
                    # gaus_fit = self.limb_darkening_function(emission_angles_to_normalized, popt_gaus["I_0"], popt_gaus["u1"], popt_gaus["u2"])
                    # sma_fit = self.limb_darkening_function(emission_angles_to_normalized, popt_sma["I_0"], popt_sma["u1"], popt_sma["u2"])
                    
                    # # plt.plot(emission_angles_to_normalized, brightness_values, label="data")
                    # # plt.plot(emission_angles_to_normalized, fit_fit, label="fit")
                    # # plt.plot(emission_angles_to_normalized, gaus_fit, label="gaus")
                    # # plt.plot(emission_angles_to_normalized, sma_fit, label="sma")
                    # # plt.legend()
                    # # plt.ylim(bottom = 0)
                    # # plt.show()
                    # fit_r2 = r2_score(brightness_values, fit_fit)
                    # gauss_r2 = r2_score(brightness_values, gaus_fit)
                    # sma_r2 = r2_score(brightness_values, sma_fit)
                    
                    # if gauss_r2 > fit_r2 and gauss_r2 > sma_r2:
                    #     optimal_popt, optimal_pcov = popt_gaus, pcov_gaus
                    # elif sma_r2 > fit_r2 and sma_r2 > gauss_r2:
                    #     optimal_popt, optimal_pcov = popt_sma, pcov_sma
                    # else:
                    #     optimal_popt, optimal_pcov = standard_popt, standard_pcov

                    # data[wave_band][slant]["meta"]["processing"]["fitted"] = True
                    # data[wave_band][slant]["fit"] = {
                        
                    # "optimal_fit" : {"fit_params": optimal_popt, "covariance_matrix": optimal_pcov, "limb_darkening_function": self.limb_darkening_function_name},
                    # "standard_fit" : {"fit_params": optimal_popt, "covariance_matrix": optimal_pcov, "limb_darkening_function": self.limb_darkening_function_name},
                    # "gaussian_fit" : {"fit_params": optimal_popt, "covariance_matrix": optimal_pcov, "limb_darkening_function": self.limb_darkening_function_name},
                    # "moving_average_fit" : {"fit_params": optimal_popt, "covariance_matrix": optimal_pcov, "limb_darkening_function": self.limb_darkening_function_name},

                    # }
                # data[wave_band][slant]["pixel_indices"], data[wave_band][slant]["pixel_distances"], data[wave_band][slant]["emission_angles"], data[wave_band][slant]["brightness_values"]
            for slant, slant_data in wave_data.items():
                if bad:
                    data[wave_band][slant]["meta"]["processing"]["fitted"] = None
            time_spent = np.around(time.time() - self.cube_start_time, 3)
            percentage_completed = (index + 1) / leng
            total_time_left = time_spent / percentage_completed - time_spent

            print("Finished fitting", wave_band, "| Spent", time_spent, " so far | expected time left:",
                np.around(total_time_left, 2), "| total time for cube :",  np.around(total_time_left + time_spent, 3), end="\r")
        print()
        return data

    def fit_all(self, fit_types: str = "all"):
        data = self.get_filtered_data()
        force_write = (SETTINGS["processing"]["clear_cache"]
                       or SETTINGS["processing"]["redo_fitting"])
        appended_data = False
        cube_count = len(data)
        self.start_time = time.time()
        self.fit_types = fit_types
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
            data[cube_name] = self.fit(cube_data, cube_name)

        # Calculate expected time left based on the average time per cube
            check_if_exists_or_write(join_strings(
                self.save_dir, cube_name + ".pkl"), data=data[cube_name], save=True, force_write=True)
            
            time_spent = np.around(time.time() - self.cube_start_time, 3)
            percentage_completed = (index + 1) / cube_count
            total_time_left = time_spent / percentage_completed - time_spent
            print("Cube", index + 1,"of", cube_count , "| Total time for cube:", time_spent, "seconds | Total Expected time left:",
                 np.around(total_time_left,2), "seconds", "| Total time spent:", np.around(time.time() - self.start_time, 3), "seconds")        
        # if os.path.exists(join_strings(self.save_dir, SETTINGS["paths"]["cumulative_sorted_path"])) and appended_data:
        #     print("Since fitted data already exists, but new data has been appended ->")
        #     check_if_exists_or_write(join_strings(
        #         self.save_dir, SETTINGS["paths"]["cumulative_fitted_path"]), data=data, save=True, force_write=True, verbose=True)
        # else:
        #     check_if_exists_or_write(join_strings(
        #         self.save_dir, SETTINGS["paths"]["cumulative_fitted_path"]), data=data, save=True, force_write=True, verbose=True)
        if (os.path.exists(join_strings(self.save_dir, SETTINGS["paths"]["cumulative_fitted_path"])) and appended_data):
            print("Fitted data already exists, but new data has been appended")
            check_if_exists_or_write(join_strings(self.save_dir, SETTINGS["paths"]["cumulative_fitted_path"]), data = data, save=True, force_write=True)
        elif force_write:
            check_if_exists_or_write(join_strings(self.save_dir, SETTINGS["paths"]["cumulative_fitted_path"]), data = data, save=True, force_write=True)
        else:
            print("Fitted not changed since last run. No changes to save...")