import os
import os.path as path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

import numpy as np
import pandas as pd
import csv
from PIL import Image, ImageStat
from scipy.optimize import curve_fit
import time
import math
from sklearn.metrics import r2_score
import PIL
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq
from scipy.stats import linregress
import pickle
import json
import pyvims

# flyby directory location
# name of csv containing all data (inside flyby folder)
# name of flyby image data (inside flyby folder)
# save loation (default to flyby data)
import numpy as np
import cv2

# surface_windows = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True]
# surface_windows = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True,
#                    True, True, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True]
# ir_surface_windows = [99, 100, 106, 107, 108, 109, 119, 120, 121, 122, 135, 136, 137, 138, 139, 140, 141, 142, 163, 164,
#                       165, 166, 167, 206, 207, 210, 211, 212, 213, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352]

from get_settings import join_strings, check_if_exists_or_write, SETTINGS


class polar_profile:
    def __init__(self, north_orientation, distance_array: np.ndarray = None, cube_center: list = None, figures: bool = None,):
        self = self
        self.figures = {
            "original_image": False,
            "emission_heatmap": False,
            "emission_heatmap_overlay": False,
            "distance_from_center": False,
            "show_slant": False,
            "plot_polar": False,
            "full_cube_slant": False,
            "persist_figures": False,
        }
        self.figure_keys = {
            key: index for index, (key, value) in enumerate(self.figures.items())
        }
        self.saved_figures = {}
        if distance_array is not None:
            self.distance_array = distance_array
        else:
            self.distance_array = None
        if cube_center is not None:
            self.cube_center = cube_center
        else:
            self.cube_center = None
            
        self.north_orientation = north_orientation
    def figure_options(self):
        print(
            "\nCurrent Figure Setttings:\n",
            *[str(val[0]) + " = " + str(val[1])
              for val in self.figures.items()],
            "\n\n",
            sep="\n"
        )
        return self.figures

    def show(self, force=False, duration: float = None):
        if duration is not None:
            plt.pause(duration)
            plt.clf()
            return
        if force:
            plt.pause(10)
            return
        options = self.figures["persist_figures"]
        if type(options) == int:
            plt.pause(options)
            plt.clf()
            plt.close()
        elif options == "wait":
            plt.waitforbuttonpress()
            plt.close()
        elif options == "wait_till_end":
            return
        elif options == True:
            plt.show()
        elif options == False:
            return None
        else:
            plt.pause(2)
            plt.clf()
            plt.close()

    def equirectangular_projection(self, cube, index):
        # take image and apply cylindrical projection
        # after cylindrical projection, remove extraneous longitude data
        proj = pyvims.projections.equirectangular.equi_cube(cube, index, 3)
        return proj

    def find_center_of_cube(self, cube):
        flattened_array = cube.eme.flatten()

        lowest_indexes = np.argpartition(flattened_array, 5)[:4]
        lowest_indexes_2d = np.unravel_index(lowest_indexes, cube.eme.shape)

        lowest_index = np.argmin(cube.eme)
        lowest_index_2d = np.unravel_index(lowest_index, cube.eme.shape)

        center = [lowest_index_2d, np.mean(lowest_indexes_2d, axis=1)]
        center = np.mean(center, axis=0)
        center_point = [int(np.round(center[0])), int(np.round(center[1]))]

        if self.figures["emission_heatmap"]:
            fig = plt.figure(self.figure_keys["emission_heatmap"])
            plt.imshow(cube.eme)
            plt.scatter(lowest_indexes_2d[1], lowest_indexes_2d[0], s=4)
            plt.scatter(lowest_index_2d[1], lowest_index_2d[0], s=4)
            plt.scatter(center_point[1], center_point[0], c="r")
            self.show()
        if self.figures["emission_heatmap_overlay"]:
            fig = plt.figure(self.figure_keys["emission_heatmap_overlay"])
            plt.imshow(cube[int(cube.bands[68])])
            plt.scatter(lowest_indexes_2d[1], lowest_indexes_2d[0], s=4)
            plt.scatter(lowest_index_2d[1], lowest_index_2d[0], s=4)
            plt.scatter(center_point[1], center_point[0], c="r")
            self.show()
            self.saved_figures["emission_heatmap_overlay"] = fig
        return center, center_point

    def distance_from_center_map(self, cube):
        # find distance from center
        center, center_point = self.find_center_of_cube(cube)
        ret = np.empty(cube.eme.shape)
        for y in range(cube.eme.shape[0]):
            for x in range(cube.eme.shape[1]):
                ret[y, x] = np.sqrt(
                    (x - center_point[1]) ** 2 + (y - center_point[0]) ** 2
                )
        if self.figures["distance_from_center"]:
            fig = plt.figure(self.figure_keys["distance_from_center"])
            plt.imshow(ret)
            self.show()
            self.saved_figures["distance_from_center"] = fig
        return ret, center, center_point

    def slanted_cross_section(self, cube, band, degree):
        """
            Get the cross section of the cube at a given angle relative to the center point (top is 0 degrees, going clockwise)
            We return the distance from the center point, the emission angle, the brightness, and the pixel indices of the cross section

            returns:
                distances: the distance from the center point
                emission_angles: the emission angle of the cross section
                brightness_values: the brightness values of the cross section
                pixel_indices: the pixel indices of the cross section
        """
        angle_rad = np.radians(degree)
        center_point = np.around(self.cube_center)
        start_x = center_point[1]
        start_y = center_point[0]
        dist = np.sqrt(
            (self.distance_array.shape[0] - center_point[0]) ** 2
            + (self.distance_array.shape[1] - center_point[1]) ** 2
        )
        endpoint_x = start_x + np.sin(angle_rad) * dist
        endpoint_y = start_y - np.cos(angle_rad) * dist
        mask = np.zeros(self.distance_array.shape)
        cv2.line(
            mask, (int(start_x), int(start_y)), (int(
                endpoint_x), int(endpoint_y)), 1, 1
        )
        line_indexes = np.where(mask == 1)
        line_indexes = np.array(line_indexes).T
        if self.figures["show_slant"]:
            fig = plt.figure(self.figure_keys["show_slant"])
            plt.scatter(line_indexes[:, 1], line_indexes[:, 0], marker="s")
            img = cube[band].copy()
            img[line_indexes[:, 0], line_indexes[:, 1]] = 0
            plt.imshow(img)
            self.show()
            self.saved_figures["show_slant_" + str(degree)] = fig
        try:
            distances = [self.distance_array[x, y] for x, y in line_indexes]
            distances = np.array(distances)
        except:
            print(line_indexes)
        brightness_values = [cube[band][x, y] for x, y in line_indexes]
        # sort values by distance
        pairs = list(zip(distances, brightness_values, line_indexes))

        # Sort the pairs based on distances
        sorted_pairs = sorted(pairs, key=lambda x: x[0])

        # Unpack the sorted pairs into separate arrays
        sorted_distances, sorted_brightness_values, pixel_indices = zip(
            *sorted_pairs)
        emission_angles = [cube.eme[x, y] for x, y in line_indexes]
        if self.figures["plot_polar"]:
            fig = plt.figure(self.figure_keys["plot_polar"])
            plt.plot(sorted_distances, sorted_brightness_values)
            self.show()
            self.saved_figures["plot_polar_" + str(degree)] = fig
        return distances, emission_angles, brightness_values, pixel_indices

    def remove_duplicates(self, x, y):
        data_dict = {}
        for x_val, y_val in zip(x, y):
            if x_val in data_dict:
                data_dict[x_val].append(y_val)
            else:
                data_dict[x_val] = [y_val]

        averaged_x = []
        averaged_y = []
        for x_val, y_vals in data_dict.items():
            averaged_x.append(x_val)
            averaged_y.append(sum(y_vals) / len(y_vals))
        return np.array(averaged_x), np.array(averaged_y)

    def complete_image_analysis_from_cube(
        self,
        cube,
        band: int,
        band_index: int,
    ):
        self.cube_name = cube.img_id
        slants = [0, 15, 30, 45, 60, 90, 120, 135, 150,
            180, 210, 225, 240, 270, 300, 315, 330]

        degrees = self.north_orientation + np.array(slants)
        cmap = matplotlib.cm.get_cmap("rainbow")
        fits = {}

        for degree in degrees:
            (
                pixel_distance,
                emission_angles,
                brightness,
                pixel_indices,
            ) = self.slanted_cross_section(cube, band, degree)
            fits[degree] = (pixel_distance, emission_angles,
                            brightness, pixel_indices)

        # if self.figures["full_cube_slant"]:
        #     fig = plt.figure(self.figure_keys["full_cube_slant"])
        #     for index, (pixel_distance, emission_angles, brightness, pixel_indices) in enumerate(fits.values()):
        #         plt.plot(emission_angles, brightness, label=str(degrees[index]), color=cmap(
        #             index/len(degrees)*0.75))
        #     plt.title("Cross sections of " + cube.flyby.name +
        #               " at " + str(cube.w[band_index]) + " µm")
        #     plt.xlabel("Distance from center (px)")
        #     plt.ylabel("Brightness")
        #     plt.legend()
        #     self.show()
        # self.saved_figures["full_cube_slant"] = fig

        # for index, (pixel_distance, emission_angles, brightness, pixel_indices) in enumerate(fits.values()):
        #     pixel_distance, emission_angles = self.remove_duplicates(pixel_distance, emission_angles)
        #     sorted_indices = np.argsort(pixel_distance)
        #     sorted_x = pixel_distance[sorted_indices]
        #     sorted_y = emission_angles[sorted_indices]
        #     csx = PchipInterpolator(sorted_x, sorted_y)
        #     xs = np.linspace(np.min(pixel_distance), np.max(pixel_distance), 1000)
        #     plt.plot(xs, gaussian_filter([csx(xss) for xss in xs], sigma=4), label=str(
        #         degrees[index]), color=cmap(index/len(degrees)*0.75))
        # plt.title("Smoothed cross sections of " + cube.flyby.name +
        #           " at " + str(cube.w[band_index]) + " µm")
        # plt.xlabel("Distance from center (px)")
        # plt.ylabel("Brightness")
        # plt.legend()
        # self.show()
        if self.figures["full_cube_slant"]:
            for index, (
                pixel_distance,
                emission_angles,
                brightness,
                pixel_indices,
            ) in enumerate(fits.values()):
                # emission_angles, brightness = self.remove_duplicates(emission_angles, brightness)
                # sorted_indices = np.argsort(emission_angles)
                # sorted_x = emission_angles[sorted_indices]
                # sorted_y = brightness[sorted_indices]
                # csx = PchipInterpolator(sorted_x, sorted_y)
                # xs = np.linspace(np.min(pixel_distance), np.max(pixel_distance), 1000)
                # plt.plot(xs, gaussian_filter([csx(xss) for xss in xs], sigma=4), label=str(degrees[index]), color=cmap(index/len(degrees)*0.75))
                sorted_indices = np.argsort(emission_angles)
                pairs = list(zip(emission_angles, brightness))

                # Sort the pairs based on distances
                sorted_pairs = sorted(pairs, key=lambda x: x[0])

                # Unpack the sorted pairs into separate arrays
                emission_angles, brightness = zip(*sorted_pairs)
                plt.plot(
                    emission_angles,
                    brightness,
                    label=str(degrees[index]),
                    color=cmap(index / len(degrees) * 0.75),
                )

            plt.title(
                "Smoothed cross sections of "
                + cube.flyby.name
                + " at "
                + str(cube.w[band_index])
                + " µm"
            )
            plt.xlabel("Eme")
            plt.ylabel("Brightness")
            plt.legend()
            self.show()

        if self.figures["original_image"]:
            fig = plt.figure(self.figure_keys["original_image"])
            plt.imshow(cube[band], cmap="gray")
            plt.title(
                "Cube from "
                + cube.flyby.name
                + " at "
                + str(cube.w[band_index])
                + " µm"
            )
            self.show()
            self.saved_figures["original_image"] = fig

        if self.figures["persist_figures"] == "wait_till_end":
            self.show(force=True)
        plt.close("all")
        return [fits, slants]


class complete_cube:
    def __init__(self, cube_name: str):
        # if data_save_folder_name is None:
        #     save_location = join_strings(
        #         parent_directory, cube_subdirectory, "analysis/limb/")
        # else:
        #     save_location = join_strings(
        #         parent_directory, cube_subdirectory, data_save_folder_name, "limb/")
        self.cube_name = cube_name
        # self.result_data_base_loc = save_location
        # self.parent_directory = parent_directory
        # self.cwd = join_strings(self.parent_directory, cube_subdirectory)

    def save_data_as_json(self, data: dict, save_location: str, overwrite: bool = None):
        if overwrite is None:
            overwrite = self.clear_cache
        if overwrite:
            with open(save_location, "w") as outfile:
                json.dump(data, outfile)
            return "not implemented yet"

    def find_north_and_south(self, cube):
        objectx = polar_profile(0)
        distance_array, center_of_cube, center_point = objectx.distance_from_center_map(
            cube)

        center_of_cube, center_point = objectx.find_center_of_cube(
            cube)
        angles = []
        brightnesses = cube.lat.flatten()
        for y in range(cube.lat.shape[0]):
            for x in range(cube.lat.shape[1]):
                angles.append(
                    int(
                        np.degrees(np.arctan2(
                            x - center_point[1], center_point[0] - y))
                    )
                )
        actual_angles = sorted(set(angles))
        br = [
            np.mean(
                [brightnesses[index]
                    for index, a in enumerate(angles) if a == angle]
            )
            for angle in actual_angles
        ]
        min_angle = actual_angles[np.argmin(br)]
        if min_angle < 0:
            min_angle += 360
        max_angle = actual_angles[np.argmax(br)]
        if max_angle < 0:
            max_angle += 360
        calculated_rots = np.array([min_angle, max_angle])
        if min_angle > max_angle:
            rot_angle = np.mean([min_angle, max_angle]) - 90
        else:
            rot_angle = np.mean([min_angle, max_angle]) + 90
        if rot_angle > 180:
            rot_angle -= 360
        return rot_angle, distance_array, center_of_cube, center_point

    def run_analysis(
        self,
        cube,
        north_orientation,
        distance_array,
        center_of_cube,
        band,
        band_index,
        key,
    ):
        start = time.time()
        analysis = polar_profile(north_orientation = north_orientation, distance_array = distance_array, cube_center = center_of_cube)
        data = analysis.complete_image_analysis_from_cube(
            cube,
            int(band),
            band_index=band_index,
        )

        data.append(
            {
                "flyby": cube.flyby.name,
                "cube": cube.img_id,
                "band": cube.w[band_index],
                "center_of_cube": center_of_cube,
                "north_orientation": north_orientation,
            }
        )

        end = time.time()
        total_time = end - self.start_time
        percentage = (band - self.starting_band) / (
            self.total_bands - self.starting_band
        )

        print(
            key,
            "took",
            end - start,
            "seconds.",
            "expected time left:",
            total_time / percentage - total_time,
            "seconds",
        )
        return data, analysis

    def analyze_dataset(self, cube_root: str, cube: str = None, force=False):
        if cube is None:
            print("since no cube was provided, defaulting to cube", self.cube_name)
            cube = self.cube_name
            self.cube_vis = pyvims.VIMS(cube, self.cwd, channel="vis")
            self.cube_ir = pyvims.VIMS(cube, self.cwd, channel="ir")
        else:
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
        self.total_bands = self.cube_vis.bands.shape[0] + \
            self.cube_ir.bands.shape[0]
        self.start_time = time.time()
        self.starting_band = -1
        (
            north_orientation,
            distance_array,
            center_of_cube,
            center_point,
        ) = self.find_north_and_south(self.cube_vis)
        datas = {}
        objects = {}
        for band in self.cube_vis.bands:
            band_index = int(band) - 1
            # if surface_windows[band_index]:
            #     continue
            key = str(self.cube_vis.w[band_index]) + "µm_" + str(int(band))
            if self.starting_band == -1:
                self.starting_band = int(band)
            data, cube_object = self.run_analysis(self.cube_vis, north_orientation=north_orientation, distance_array=distance_array, center_of_cube=center_of_cube, band=band, band_index=band_index, key=key)
            datas[key] = data
            objects[key] = cube_object
        for band in self.cube_ir.bands:
            band_index = int(band) - 97
            # if int(band) in ir_surface_windows:
            #     continue
            key = str(self.cube_ir.w[band_index]) + "µm_" + str(int(band))
            if self.starting_band == -1:
                self.starting_band = int(band)
            data, cube_object = self.run_analysis(
                self.cube_ir, north_orientation=north_orientation, distance_array=distance_array, center_of_cube=center_of_cube, band=band, band_index=band_index, key=key)
            datas[key] = data
            objects[key] = cube_object
        return datas, objects


class analyze_complete_dataset:
    def __init__(self, cubes_location: str = None) -> None:
        if cubes_location is None:
            cubes_location = join_strings(
                SETTINGS["paths"]["parent_data_path"],
                SETTINGS["paths"]["cube_sub_path"],
            )
        self.cubes_location = cubes_location
        folders = os.listdir(self.cubes_location)
        self.cubes = [
            folder
            for folder in folders
            if folder.startswith("C") and folder[-2] == ("_")
        ]
        self.cubes.sort()

        # self.figures = {
        # }
        # self.figure_keys = {key: index for index,
        #                     (key, value) in enumerate(self.figures.items())}

    # def figure_options(self):
    #     print("\nCurrent Figure Setttings:\n", *[str(val[0]) + " = " + str(
    #         val[1]) for val in self.figures.items()], "\n\n", sep="\n")
    #     return self.figures

    def complete_dataset_analysis(self):
        all_data = {}
        # all_objects = {}
        for cub in self.cubes:
            image = complete_cube(cub)
            datas, objects = image.analyze_dataset(cube_root=join_strings(self.cubes_location, cub), cube=cub, force=SETTINGS["processing"]["clear_cache"])
            fit_cube = join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["analysis_sub_path"], "limb/")
            if SETTINGS["processing"]["clear_cache"] == True:
                check_if_exists_or_write(
                    cub + ".pkl",
                    base=fit_cube,
                    prefix="",
                    save=True,
                    data=datas,
                    force_write=True,
                    verbose=True,
                )
                # check_if_exists_or_write(
                #     cub + "_objects.pkl",
                #     base=fit_cube,
                #     prefix="",
                #     save=True,
                #     data=objects,
                #     force_write=True,
                #     verbose=True,
                # )
            all_data[cub] = datas
            # all_objects[cub] = objects
        if SETTINGS["processing"]["clear_cache"] == True:
            # check_if_exists_or_write(
            #     "combined_objects.pkl",
            #     base=self.cubes_location,
            #     prefix="analysis/limb",
            #     save=True,
            #     data=all_objects,
            #     force_write=True,
            #     verbose=True,
            # )         
            
            check_if_exists_or_write(
                SETTINGS["paths"]["cumulative_data_path"],
                base=join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["analysis_sub_path"], "limb/"),
                save=True,
                data=all_data,
                force_write=True,
                verbose=True,
            )

    # def sort_and_remove_outliers(self, eme, brightness, threshold=0.5):
    #     zi = zip(eme, brightness)
    #     zi = sorted(zi, key=lambda x: x[0])
    #     eme, brightness = zip(*zi)
    #     eme = np.array(eme)
    #     brightness = np.array(brightness)
    #     mask = eme <= 90-threshold

    #     # Apply the mask to both x and y arrays
    #     filtered_x = eme[mask]
    #     filtered_y = brightness[mask]
    #     return filtered_x, filtered_y
    # def timeline_figure(self, band: int = None):
    #     all_fits = {}
    #     fig, axs = plt.subplots(4, 3, figsize=(12, 8))
    #     axs = axs.flatten()
    #     plt.title(str(band))
    #     for index, cub in enumerate(self.cubes):
    #         cube = join_strings(self.cubes_location, cub, "analysis/limb/fits")
    #         bands = os.listdir(cube)
    #         band_file = [band_f for band_f in bands if band_f.endswith(
    #             '_' + str(band) + ".pkl")][0]
    #         with open(join_strings(cube, band_file), 'rb') as handle:
    #             all_fits[cub] = pickle.load(handle)
    #     quantity = len(all_fits)
    #     cmap = matplotlib.colormaps.get_cmap('rainbow')
    #     for index, (cube, fit) in enumerate(all_fits.items()):
    #         list_keys = list(fit[0].keys())

    #         for ind, deg in enumerate(list_keys):
    #             emission, brightness = self.sort_and_remove_outliers(fit[0][deg][1], fit[0][deg][2])
    #             if ind == 0:
    #                 axs[index].plot(emission, brightness, label = fit[2]["flyby"] + " " + str(deg),  color= cmap(ind/len(list_keys)*0.75))
    #             else:
    #                 axs[index].plot(emission, brightness, label = fit[2]["flyby"] + " " +  str(deg), color= cmap(ind/len(list_keys)*0.75))
    #         axs[index].set_xlim(0, 95)
    #         axs[index].legend(fontsize=3)
    #     fig.tight_layout()
    #     plt.show()
    #     return all_fits

    # def generate_figures(self, figures: dict = None):
    #     for band in range(1, 352, 10):
    #         # if band < 97 and surface_windows[band-1] == True:
    #         #     continue
    #         # elif band > 96 and band in ir_surface_windows:
    #         #     continue
    #         self.timeline_figure(band)


analyze = analyze_complete_dataset()
analyze.complete_dataset_analysis()

# analyze.timeline_figure(band = 1)
