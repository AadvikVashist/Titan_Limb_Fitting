from .polar_profile import analyze_complete_dataset, destripe_VIMS_V
from .north_south_asymmetry import nsa
from settings.get_settings import join_strings, check_if_exists_or_write, SETTINGS, get_cumulative_filename
import re
import time
import os
import numpy as np
import pyvims
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime, timezone

DATES = [1980 + 11/12,1981 + 8/12,  1990+8/12, 1992 + 8/12, 1994 + 10/12, 1995 + 8/12, 1997 + 11/12, 2002 + 12/12, 2003 + 12/12, 2004 + 10/12, 2004 + 10/12, 2004 + 12/12, 2005 + 2/12, 2005+ 12/12, 2005 + 10/12, 2007 + 5/12, 2007 + 12/12, 2007+ 5/12, 2009 + 8 / 12, 2009 + 10/ 12, 2010 + 4/12, 2011 + 12/12, 2012 + 7/12, 2013 + 10/12, 2017 + 6/12, 2017+7/12]
DATES = np.array(DATES) - 1/12

# Boundary Latitudes
BOUNDARY_LATS = [-5.5, -5.5, 32, 20, 15, 15, 10, -20, -20, -12, -8, -8, -8, -8, -12, -8, -8, -12, -12, -12, -12, -10, -10, -5, 10, 10]

# Errors of Boundary Latitudes
BOUNDARY_ERRORS = [1, 1, 10, 10, 5, 5, 15, 10, 5, 3, 2, 2, 2, 2, 3, 2, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4]




def cyclical_curve(x, a, b, c):
    return a * np.sin(2 * np.pi * x / 29.5 + b) + c

class insert_nsa:
    def __init__(self):
        self.save_dir = join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["nsa_data_sub_path"])
        cubes_location = join_strings(
                SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"],
    )
        self.cubes_location = cubes_location
        folders = os.listdir(self.cubes_location)
        self.cubes = [
            folder
            for folder in folders
            if folder.startswith("C") and folder[-2] == ("_")
        ]
        self.cubes.sort()
    def get_cube_data(self):

        cubs = os.listdir(self.cubes_location)
        cubs = [cub for cub in cubs if re.fullmatch(r'C.*_.*', cub) is not None]
        return cubs
    def process_nsa(self, cube_root: str, cube: str = None, force=False):
        # if cube is None:
        #     self.cube_vis = pyvims.VIMS(cube, self.cwd, channel="vis")
        #     self.cube_ir = pyvims.VIMS(cube, self.cwd, channel="ir")
        cube_path = join_strings(cube_root, cube)
        if not os.path.exists(cube_path + "_vis.cub"):
            print("vis cube not found, checking dir", cube_path + "_vis.cub")
            return None
        if not os.path.exists(cube_path + "_ir.cub"):
            print("ir cube not found, checking dir", cube_path + "_ir.cub")
            return None
        self.cube_vis = pyvims.VIMS(
            cube + "_vis.cub", cube_root, channel="vis")
        self.cube_ir = pyvims.VIMS(
            cube + "_ir.cub", cube_root, channel="ir")
        nsa_calc = nsa.nsa_analysis(cube_vis=self.cube_vis, cube_ir=self.cube_ir, cube_name=cube)
        nsa_ret = nsa_calc.analyze_all_wavelengths()
        
        # data["meta"]["nsa"] = nsa_ret
        # for index, (wave_band, wave_data) in enumerate(data.items()):
        #     if "µm_" not in wave_band:
                # continue
            
            # for slant, slant_data in wave_data.items():
            #     x=0
                #calculate how nsa affects slant (make a mask list)
        # print("Finished nsa calculations", wave_band, "| Spent", np.around(time.time() - self.cube_start_time, 3), "| expected time left:", np.around((time.time() - self.cube_start_time) * (leng - index - 1),2), end="\r")
        return nsa_ret
    
    def fitting_cycle(self):
        smooth_dates = np.linspace(DATES.min(), DATES.max(), 1000)


        # Defining a cyclical curve function with a period of 29.5 years

        

        # Curve fitting
        weights = 1 / np.array(BOUNDARY_ERRORS)
        params_with_errors, _ = curve_fit(cyclical_curve, DATES, BOUNDARY_LATS, sigma=weights)

        # Generating a smooth curve for the new fit
        fitted_curve_with_errors = cyclical_curve(smooth_dates, *params_with_errors)

        # Plotting the data, the fitted curve, and error bars
        plt.figure(figsize=(10, 5))
        plt.errorbar(DATES, BOUNDARY_LATS, yerr=BOUNDARY_ERRORS, fmt='o', color='blue', label='Data Points with Error Bars')
        plt.plot(smooth_dates, fitted_curve_with_errors, color='red', label='Fitted Cyclical Curve with Error Bars')
        plt.xlabel('Year')
        plt.ylabel('Boundary Latitude')
        plt.title('Cyclical Curve Fitting with Error Bars and a Period of 29.5 Years')
        plt.legend()
        plt.show()
        return params_with_errors
    def process_cube(self, fitting_params, cube_root: str, cube: str = None):
        cube_path = join_strings(cube_root, cube)
        if not os.path.exists(cube_path + "_vis.cub"):
            print("vis cube not found, checking dir", cube_path + "_vis.cub")
            return None
        if not os.path.exists(cube_path + "_ir.cub"):
            print("ir cube not found, checking dir", cube_path + "_ir.cub")
            return None
        cube_vis = pyvims.VIMS(
            cube + "_vis.cub", cube_root, channel="vis")
        cube_ir = pyvims.VIMS(
            cube + "_ir.cub", cube_root, channel="ir")
        cube_time = np.mean([self.get_time(cube_vis), self.get_time(cube_ir)])
        latitude = cyclical_curve(cube_time, *fitting_params)
        return latitude

    def get_time(self, cube):
        dt =  cube.time
        year = dt.year
        start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
        
        # Total number of days in the year
        year_length = 365
        
        # Number of days passed in the year
        days_passed = (dt - start_of_year).days
        
        # Converting to decimal year
        decimal_year = year + days_passed / year_length
        
        return decimal_year
    def insert_nsa_data_in_all(self):
        force_write = (SETTINGS["processing"]["clear_cache"] or SETTINGS["processing"]["redo_nsa_calculations"])
        
        if os.path.exists(self.save_dir) and all([cub + ".pkl" in os.listdir(self.save_dir) for cub in self.cubes]) and os.path.exists(join_strings(self.save_dir, get_cumulative_filename("nsa_data_sub_path"))) and not force_write:
            print("Data already sorted")
            return

        data = self.get_cube_data()
        # for cube_name, cube_data in data.items():
        appended_data = False
        cube_count = len(data)
        cum_data = {}
        
        fitting_params = self.fitting_cycle()
        for index, cube_name in enumerate(data):      
            if os.path.exists(join_strings(self.save_dir, cube_name + ".pkl")) and not force_write:
                print("NSA data already exists. Skipping...")
                cum_data[cube_name] = check_if_exists_or_write(join_strings(self.save_dir, cube_name + ".pkl"), save=False)
                print(index,cum_data[cube_name]["nsa_latitude"])
                continue
            elif not force_write:
                appended_data = True
            
            latitude = self.process_cube(fitting_params= fitting_params, cube_root=join_strings(self.cubes_location, cube_name), cube=cube_name)
            
            cum_data[cube_name] = {"nsa_latitude": latitude}
            check_if_exists_or_write(join_strings(self.save_dir, cube_name + ".pkl"), data=cum_data[cube_name], save=True, force_write=True)
            
            

        if (os.path.exists(join_strings(self.save_dir, get_cumulative_filename("nsa_data_sub_path"))) and appended_data):
            print("NSA data already exists, but new data has been appended")
            check_if_exists_or_write(join_strings(self.save_dir,get_cumulative_filename("nsa_data_sub_path")), data = data, save=True, force_write=True)
        elif force_write:
            check_if_exists_or_write(join_strings(self.save_dir, get_cumulative_filename("nsa_data_sub_path")), data = data, save=True, force_write=True)
        elif not os.path.exists(join_strings(self.save_dir, get_cumulative_filename("nsa_data_sub_path"))):
            print("NSA data not found. Creating new file...")
            check_if_exists_or_write(join_strings(self.save_dir, get_cumulative_filename("nsa_data_sub_path")), data = data, save=True, force_write=True)
        else:
            print("NSA data not changed since last run. No changes to save...")
            
    # def insert_nsa_data_in_all(self):
    #     force_write = (SETTINGS["processing"]["clear_cache"] or SETTINGS["processing"]["redo_nsa_calculations"])
        
    #     if os.path.exists(self.save_dir) and all([cub in os.listdir(self.save_dir) for cub in os.listdir(self.cubes_location)]) and os.path.exists(join_strings(self.save_dir, get_cumulative_filename("nsa_data_sub_path"))) and not force_write:
    #         print("Data already sorted")
    #         return

    #     data = self.get_cube_data()
    #     # for cube_name, cube_data in data.items():
    #     appended_data = False
    #     cube_count = len(data)
    #     self.start_time = time.time()
    #     cum_data = {}
        

    #     for index, cube_name in enumerate(data):      
    #         if os.path.exists(join_strings(self.save_dir, cube_name + ".pkl")) and not force_write:
    #             print("NSA data already exists. Skipping...")
    #             cum_data[cube_name] = check_if_exists_or_write(join_strings(self.save_dir, cube_name + ".pkl"), save=False)
    #             print(index,cum_data[cube_name]["nsa_latitude"])
    #             continue
    #         elif not force_write:
    #             appended_data = True
            
    #         self.cube_start_time = time.time()
    #         cum_data[cube_name] = self.process_nsa(cube_root=join_strings(self.cubes_location, cube_name), cube=cube_name, force=force_write)
    #         check_if_exists_or_write(join_strings(self.save_dir, cube_name + ".pkl"), data=cum_data[cube_name], save=True, force_write=True)
            
    #         time_spent = np.around(time.time() - self.cube_start_time, 3)
    #         percentage_completed = (index + 1) / cube_count
    #         total_time_left = time_spent / percentage_completed - time_spent
            
    #         print("Cube", index + 1,"of", cube_count , "| Total time for cube:", time_spent, "seconds | Total Expected time left:", np.around(total_time_left,2), "seconds", "| Total time spent:", np.around(time.time() - self.start_time, 3), "seconds")        
        
    #     if (os.path.exists(join_strings(self.save_dir, get_cumulative_filename("nsa_data_sub_path"))) and appended_data):
    #         print("NSA data already exists, but new data has been appended")
    #         check_if_exists_or_write(join_strings(self.save_dir,get_cumulative_filename("nsa_data_sub_path")), data = data, save=True, force_write=True)
    #     elif force_write:
    #         check_if_exists_or_write(join_strings(self.save_dir, get_cumulative_filename("nsa_data_sub_path")), data = data, save=True, force_write=True)
    #     elif not os.path.exists(join_strings(self.save_dir, get_cumulative_filename("nsa_data_sub_path"))):
    #         print("NSA data not found. Creating new file...")
    #         check_if_exists_or_write(join_strings(self.save_dir, get_cumulative_filename("nsa_data_sub_path")), data = data, save=True, force_write=True)
    #     else:
    #         print("NSA data not changed since last run. No changes to save...")
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
# class complete_cube:
#     def __init__(self, cube_name: str):
#         self.cube_name = cube_name

#     def run_analysis(
#         self,
#         cube,
#         north_orientation,
#         distance_array,
#         center_of_cube,
#         band,
#         band_index,
#         key,
#     ):
#         start = time.time()
#         analysis = polar_profile(north_orientation=north_orientation,
#                                  distance_array=distance_array, cube_center=center_of_cube)
#         data = analysis.complete_image_analysis_from_cube(
#             cube, int(band), band_index=band_index)

#         end = time.time()
#         total_time = end - self.start_time
#         percentage = (band - self.starting_band) / \
#             (self.total_bands - self.starting_band)

#         print(key, "took", end - start, "seconds.", "expected time left:",
#               total_time / percentage - total_time, "seconds", end="\r")
#         # print()
#         return data

#     def analyze_dataset(self, cube_root: str, cube: str = None, force=False):
#         if cube is None:
#             self.cube_vis = pyvims.VIMS(cube, self.cwd, channel="vis")
#             self.cube_ir = pyvims.VIMS(cube, self.cwd, channel="ir")
#         else:
#             cube_path = join_strings(cube_root, cube)
#             if not os.path.exists(cube_path + "_vis.cub"):
#                 print("vis cube not found, checking dir", cube_path + "_vis.cub")
#                 return None
#             if not os.path.exists(cube_path + "_ir.cub"):
#                 print("ir cube not found, checking dir", cube_path + "_ir.cub")
#                 return None
#             self.cube_vis = pyvims.VIMS(
#                 cube + "_vis.cub", cube_root, channel="vis")
#             self.cube_ir = pyvims.VIMS(
#                 cube + "_ir.cub", cube_root, channel="ir")
#         self.total_bands = self.cube_vis.bands.shape[0] + \
#             self.cube_ir.bands.shape[0]
#         self.start_time = time.time()
#         self.starting_band = -1
#         (north_orientation, distance_array, center_of_cube, center_point) = self.find_north_and_south(self.cube_vis)
#         incident_angle, incident_location = self.inc_sampling(self.cube_vis, center_point, north_orientation)
#         datas = {}
#         datas["meta"] = {
#             "cube_vis":  self.convert_cube_to_dict(self.cube_vis),
#             "cube_ir": self.convert_cube_to_dict(self.cube_ir),
#             "lowest_inc_vis": incident_angle,
#             "lowest_inc_location_vis": incident_location,
#             "north_orientation_vis": north_orientation,
#             "center_of_cube_vis": center_of_cube,
#         }
#         for band in self.cube_vis.bands:
#             band_index = int(band) - 1
#             # if surface_windows[band_index]:
#             #     continue
#             key = str(self.cube_vis.w[band_index]) + "µm_" + str(int(band))
#             if self.starting_band == -1:
#                 self.starting_band = int(band)
#             data = self.run_analysis(self.cube_vis, north_orientation=north_orientation,distance_array=distance_array, center_of_cube=center_of_cube, band=band, band_index=band_index, key=key)
#             datas[key] = data
        
        
#         (north_orientation, distance_array, center_of_cube, center_point) = self.find_north_and_south(self.cube_ir)
#         incident_angle, incident_location = self.inc_sampling(self.cube_vis, center_point, north_orientation)
#         for band in self.cube_ir.bands:
#             band_index = int(band) - 97
#             # if int(band) in ir_surface_windows:
#             #     continue
#             key = str(self.cube_ir.w[band_index]) + "µm_" + str(int(band))
#             if self.starting_band == -1:
#                 self.starting_band = int(band)
#             data = self.run_analysis(self.cube_ir, north_orientation=north_orientation, distance_array=distance_array, center_of_cube=center_of_cube, band=band, band_index=band_index, key=key)
#             datas[key] = data

#         datas["meta"]["lowest_inc_ir"] = incident_angle
#         datas["meta"]["north_orientation_ir"] = north_orientation
#         datas["meta"]["center_of_cube_ir"] = center_of_cube  
#         datas["meta"]["lowest_inc_location_ir"] = incident_location

#         #             data.append(
#         #     {
#         #         "flyby": cube.flyby.name,
#         #         "cube": cube.img_id,
#         #         "band": cube.w[band_index],
#         #         "center_of_cube": center_of_cube,
#         #         "north_orientation": north_orientation,
#         #     }
#         # )

#         return datas


