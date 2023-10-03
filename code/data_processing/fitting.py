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
import multiprocessing
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
            SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["nsa_filtered_out_sub_path"])


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
        if os.path.exists(join_strings(self.data_dir,get_cumulative_filename("nsa_filtered_out_sub_path"))):
            all_data = check_if_exists_or_write(join_strings(
                self.data_dir,get_cumulative_filename("nsa_filtered_out_sub_path")), save=False, verbose=True)
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
        param_bounds = [[0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]]
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
        
    def remove_outliers(self,x,y):
        std = np.std(y, axis=0)
        mean = np.mean(y, axis=0) - np.min(y)
        diff = np.diff(y, axis=0); diff = np.insert(diff, 0, 0, axis=0)
        # activated = True
        # ret_x = x.copy()
        # ret_y = y.copy()
        # for index in range(len(y)):
        #     if index > 1 and index < len(y) - 2:
        #         three_vals = y[index-1:index+2]
        #         std_val = np.std(three_vals) 
        #         if std_val > std:
                    
        #             y[index] = np.mean([three_vals[0], three_vals[2]])
        #             activated = True
                    
        if np.max(np.abs(diff)) > 0.5 * mean:
            plt.plot(x,y)
            plt.plot(x, diff)

            plt.show()
            
        return x, y
        
    def limb_darkening_laws(self, emission_angles, brightness_values, fit_types):
        #get the interpolated data by normalizing the emission angles, and sorting the data
        # emission_angles, brightness_values = self.remove_outliers(emission_angles, brightness_values)
        emission_angles_to_normalized = self.emission_to_normalized(emission_angles)
        pairs = list(zip(emission_angles_to_normalized, np.array(brightness_values)))


        sorted_pairs = sorted(pairs, key=lambda x: x[0])
        #the actual interpolation     
        pchip = PchipInterpolator(*zip(*sorted_pairs))
        range_val = np.max(emission_angles_to_normalized) - np.min(emission_angles_to_normalized)
        # interp_x = list(np.linspace(np.min(emission_angles_to_normalized), np.min(emission_angles_to_normalized) + 0.15 * range, 50)); interp_x.extend(np.linspace(np.min(emission_angles_to_normalized) + 0.15 * range, np.max(emission_angles_to_normalized) - 0.15 * range, 100)); interp_x.extend(np.linspace(np.max(emission_angles_to_normalized) - 0.15 * range, np.max(emission_angles_to_normalized), 50))
        interp_x = np.linspace(np.min(emission_angles_to_normalized), np.max(emission_angles_to_normalized), 200)
        interp_y = pchip(interp_x)
        

        #get the smoothed data
        sigma = 20
        window = 20
        gauss = gaussian_filter(interp_y, sigma=sigma)
        sma = simple_moving_average(interp_y, N = window)
        

        if fit_types == "all":
            fit_types = ["lin", "quad", "sqrt"]
            # fit_types = ["quad", "sqrt"]

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
                fit_r2 = np.nan
                standard_popt = None
                standard_pcov = None
            else:
                fit_fit = plot_func(emission_angles_to_normalized, *standard_popt.values())
                fit_r2 = r2_score(brightness_values, fit_fit)
            if type(gauss_pcov) == str:
                gauss_fit = None
                gauss_r2 = np.nan
                gauss_popt = None
                gauss_pcov = None
            else:
                gauss_fit = plot_func(emission_angles_to_normalized, *gauss_popt.values())
                gauss_r2 = r2_score(brightness_values, gauss_fit)
            
            if type(sma_pcov) == str:
                sma_fit = None
                sma_r2 = np.nan
                sma_popt = None
                sma_pcov = None
            else:
                sma_fit = plot_func(emission_angles_to_normalized, *sma_popt.values())
                sma_r2 = r2_score(brightness_values, sma_fit)


            return_value[fit_type] = {
            "standard_fit" : {"fit_params": standard_popt, "covariance_matrix": standard_pcov, "r2": fit_r2},
            "gaussian_fit" : {"fit_params": gauss_popt, "covariance_matrix": gauss_pcov, "r2": gauss_r2, "sigma": sigma},
            "moving_average_fit" : {"fit_params": sma_popt, "covariance_matrix": sma_pcov, "r2": sma_r2, "window": window},
            }
            list_of_r2 = [fit_r2, gauss_r2, sma_r2]
            if np.isnan(list_of_r2).all():
                optimal_selection = 0
                return_value[fit_type]["optimal_fit"] = {}  
                continue
            else:
                optimal_selection = np.nanargmax(list_of_r2)

            if type(optimal_selection) != np.int64:
                optimal_selection = 0
            if optimal_selection == 0:
                optimal_selection = "fit_r2"
            elif optimal_selection == 1:
                optimal_selection = "gauss_r2"
            elif optimal_selection == 2:
                optimal_selection = "sma_r2"
            
            if "gauss_r2" in optimal_selection:
                return_value[fit_type]["optimal_fit"] = return_value[fit_type]["gaussian_fit"]            
            elif "sma_r2" in optimal_selection:
                return_value[fit_type]["optimal_fit"] = return_value[fit_type]["moving_average_fit"]            
            else:
                return_value[fit_type]["optimal_fit"] = return_value[fit_type]["standard_fit"]            
            if type(return_value[fit_type]["optimal_fit"]) is dict and return_value[fit_type]["optimal_fit"]["fit_params"] is None:
                x = 0
        return return_value

    def fit(self, data: dict, cube_name: str = None):
        leng = len(data.keys())
        cube_vis = pyvims.VIMS(cube_name + "_vis.cub", join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube_name), channel="vis")
        cube_ir = pyvims.VIMS(cube_name + "_ir.cub",  join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],cube_name), channel="ir")
        ind_shift = 0
        for index, (wave_band, wave_data) in enumerate(data.items()):
            if "µm_" not in wave_band:
                ind_shift+=1
                continue
            if index >= 96+ind_shift:
                image = data["meta"]["cube_ir"]["bands"][index-ind_shift-96]
            else:
                image = data["meta"]["cube_vis"]["bands"][index-ind_shift]

            for slant, slant_data in wave_data.items():
                emission_angles = np.array(slant_data["emission_angles"])
                brightness_values = np.array(slant_data["brightness_values"])
                slant_b = np.array([image[pixel_index[0], pixel_index[1]] for pixel_index in data[wave_band][slant]["pixel_indices"]])
                if not np.all(slant_b == brightness_values):
                    raise ValueError("Brightness values not equal")

                if len(emission_angles) == 0:
                    data[wave_band][slant]["meta"]["processing"]["fitted"] = False
                    continue                
                ret_values = self.limb_darkening_laws(emission_angles, brightness_values, self.fit_types)
                data[wave_band][slant]["fit"] = ret_values
                if ret_values["quadratic"]["optimal_fit"] == {}:
                    #try to redo same thing with old vals
                    data[wave_band][slant]["meta"]["processing"]["fitted"] = None
                    old_eme_angles = np.array(slant_data["filtered"]["emission_angles"])
                    old_b_values = np.array(slant_data["filtered"]["brightness_values"])
                    unfiltered_ret = self.limb_darkening_laws(old_eme_angles, old_b_values, self.fit_types)
                    data[wave_band][slant]["filtered"]["fit"] = unfiltered_ret
                    if unfiltered_ret["quadratic"]["optimal_fit"] == {}:
                        data[wave_band][slant]["filtered"]["meta"]["processing"]["fitted"] = None
                    else:
                        data[wave_band][slant]["filtered"]["meta"]["processing"]["fitted"] = True

                else:
                    data[wave_band][slant]["meta"]["processing"]["fitted"] = True

                

            time_spent = np.around(time.time() - self.cube_start_time, 3)
            percentage_completed = (index + 1) / leng
            total_time_left = time_spent / percentage_completed - time_spent

            print("Finished fitting", wave_band, "| Spent", time_spent, " so far | expected time left:",
                np.around(total_time_left, 2), "| total time for cube :",  np.around(total_time_left + time_spent, 1), "                                ", end="\r")
        print()
        return data

    def multi_process_func(self, cube_data, index, cube_name, save_on_end: dict = None):
        self.ret_data[cube_name] = self.fit(cube_data, cube_name)
        check_if_exists_or_write(join_strings(self.save_dir, cube_name + ".pkl"), data=self.ret_data[cube_name], save=True, force_write=True)
        time_spent = np.around(time.time() - self.cube_start_time, 3)
        percentage_completed = (index + 1) / self.cube_count
        total_time_left = time_spent / percentage_completed - time_spent
        print("Cube", index + 1,"of", self.cube_count , "| Total time for cube:", np.around(time_spent, 1), "seconds | Total Expected time left:",
                np.around(total_time_left,2), "seconds", "| Total time spent:", np.around(time.time() - self.start_time,1), "seconds"                       )        
        if save_on_end is not None:
            self.save_cumulative(force_write=save_on_end["force_write"], appended_data=save_on_end["appended_data"])
    def save_cumulative(self, force_write: bool = False, appended_data: bool = True):

        if (os.path.exists(join_strings(self.save_dir,get_cumulative_filename("fitted_sub_path"))) and appended_data):
            print("Fitted data already exists, but new data has been appended")
            check_if_exists_or_write(join_strings(self.save_dir,get_cumulative_filename("fitted_sub_path")), data = self.ret_data, save=True, force_write=True)
        elif force_write:
            check_if_exists_or_write(join_strings(self.save_dir, get_cumulative_filename("fitted_sub_path")), data = self.ret_data, save=True, force_write=True)
        else:
            print("Fitted not changed since last run. No changes to save...")    
    def fit_some(self, start: int = 0, step: int = 3, fit_types: str = "all", multi_process: bool = False):
        self.data = self.get_filtered_data()
        force_write = (SETTINGS["processing"]["clear_cache"]
                       or SETTINGS["processing"]["redo_fitting"])
        appended_data = True
        self.cube_count = len(self.data)
        self.start_time = time.time()
        self.fit_types = fit_types
        args = []
        self.ret_data = {}
        for index, (cube_name, cube_data) in enumerate(self.data.items()):
            if index % step != start:
                print("skipping because not in step->", cube_name)
                continue
            if os.path.exists(join_strings(self.save_dir, cube_name + ".pkl")) and not force_write:
                try:
                    self.ret_data[cube_name] = check_if_exists_or_write(
                        join_strings(self.save_dir, cube_name + ".pkl"), save=False)
                    print("fitted data already exists. Skipping...")
                    continue
                except:
                    print("fitted data corrupted. Processing...")
            elif not force_write:
                appended_data = True
            self.cube_start_time = time.time()
            # only important line in this function
            if multi_process:
                args.append([cube_data, index, cube_name])
            else:
                self.multi_process_func(cube_data, index, cube_name)
        if multi_process:
            with multiprocessing.Pool(processes=3) as pool:
                pool.starmap(self.multi_process_func, args)
                pool.close()
                pool.join()  # This line ensures that all processes are done
        self.save_cumulative(force_write=force_write, appended_data=appended_data)

            
    def fit_all(self, fit_types: str = "all", multi_process: bool = False):
        self.data = self.get_filtered_data()
        force_write = (SETTINGS["processing"]["clear_cache"]
                       or SETTINGS["processing"]["redo_fitting"])
        appended_data = True
        self.cube_count = len(self.data)
        self.start_time = time.time()
        self.fit_types = fit_types
        args = []
        self.ret_data = {}
        for index, (cube_name, cube_data) in enumerate(self.data.items()):
            if os.path.exists(join_strings(self.save_dir, cube_name + ".pkl")) and not force_write:
                try:
                    self.ret_data[cube_name] = check_if_exists_or_write(
                        join_strings(self.save_dir, cube_name + ".pkl"), save=False)
                    print("fitted data already exists. Skipping...")
                    continue
                except:
                    print("fitted data corrupted. Processing...")
            elif not force_write:
                appended_data = True
            self.cube_start_time = time.time()
            # only important line in this function
            if multi_process:
                args.append([cube_data, index, cube_name])
            else:
                self.multi_process_func(cube_data, index, cube_name)

        if multi_process:
            with multiprocessing.Pool(processes=3) as pool:
                pool.starmap(self.multi_process_func, args)
                pool.close()
                pool.join()  # This line ensures that all processes are done
        self.save_cumulative(force_write=force_write, appended_data=appended_data)
