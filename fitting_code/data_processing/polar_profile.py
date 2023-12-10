from settings.get_settings import join_strings, check_if_exists_or_write, SETTINGS, get_cumulative_filename
import cv2
import pyvims
import os
import matplotlib.pyplot as plt
import matplotlib

import numpy as np

from scipy.optimize import curve_fit
import time

import multiprocessing
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from typing import Union
# Define the sawtooth function
# def sawtooth(x, amplitude, frequency, phase, offset):
#     return amplitude * (2 * (x * frequency + phase) - 2 * np.floor(x * frequency + phase + 0.5)) + offset


def triangle_wave(x, amplitude, frequency, phase, offset):
    return amplitude * np.abs(np.mod(x / (360 / frequency) + phase, 1) - 0.5) * 4 - amplitude + offset


def find_closest_distance_to_edges(circle_mask, band):
    # Check if the mask is empty or does not contain any True value
    if not circle_mask.any():
        return None

    # Identify the rows and columns indices of the True values
    rows, _ = np.where(circle_mask)

    # Calculate distances to top and bottom edges
    top_distances = []
    bottom_distances = []
    for r in rows:
        top_distance = r  # Distance to the top edge
        bottom_distance = circle_mask.shape[0] - r - 1  # Distance to the bottom edge

        # Add the minimum distance for this point
        top_distances.append(top_distance)
        bottom_distances.append(bottom_distance)
    if band == 1:
        print(min(top_distances) + min(bottom_distances))
    # if band < 2:
    #     plt.title(min(distances))
    #     plt.imshow(circle_mask)
    #     plt.show()
    # Return the minimum distance among all points
    return min(top_distances) + min(bottom_distances)



# def take_surface_measurements_and_get_surface(measurements):
#     surface = np.mean(measurements,axis = 1)
def destripe_VIMS_V(cube, band, save = False, actual_surface = None):
    start_time = time.time()
    dicts = {}
    cube_band = cube[int(band)]
    surface = cube.ground
    img = cube_band - np.min(cube_band)
    img = img / np.max(img) * 255
    img = np.asarray(img, dtype=np.uint8)  # Use np.uint8 instead of np.int8 
    now_time = time.time(); dicts["convert_img"] = now_time - start_time; start_time = now_time
    
    # if actual_surface == None:
    #     detected_circles = cv2.HoughCircles(cv2.GaussianBlur(img, (5, 5), 0),
    #                                         cv2.HOUGH_GRADIENT, 1, 20,
    #                                         param1=50, param2=30,
    #                                         minRadius=int(np.sqrt(np.sum(surface)/np.pi)*0.85))
    #     now_time = time.time(); dicts["hough"] = now_time - start_time; start_time = now_time
    #     # Draw the circles that were detected
    #     if detected_circles is not None:
    #         detected_circles = detected_circles[0][0]
    #         center_x, center_y, radius = np.round(detected_circles); center_x = int(center_x); center_y = int(center_y); radius = int(radius)
    #         height, width = img.shape[:2]
    #         mask = np.zeros((height, width), dtype=np.uint8)

    #         # Draw a filled circle on the mask
    #         cv2.circle(mask, (center_x, center_y), radius, 255, thickness=-1)

    #         # Extract pixel values from the image using the mask
    #         actual_surface = mask == 255

            
    #     else:

    #         actual_surface = cv2.dilate(surface.astype(np.uint8),
    #                         np.ones((3, 3), np.uint8), iterations=3)
        
    now_time = time.time(); dicts["actual surface"] = now_time - start_time; start_time = now_time
    img_size = np.mean(img.shape) 
    iterations = np.round(img_size / 15).astype(int)
    actual_surface = cv2.dilate(surface.astype(np.uint8),
                        np.ones((3, 3), np.uint8), iterations=iterations)
    now_time = time.time(); dicts["dilate"] = now_time - start_time; start_time = now_time
    
    min_dist = find_closest_distance_to_edges(actual_surface, band)
    if min_dist < 4:
        return cube_band
    col_no_surface = np.where(actual_surface, np.nan, cube_band)

    col_avg = []
    sub_col = []
    sub_index = [ ]
    for column in range(col_no_surface.shape[1]):
        if np.sum(~np.isnan(col_no_surface[:,column])) < 3:
            col_avg.append(np.nan)
        else:
            avg = np.nanmean(col_no_surface[:,column]) * 0.3 + np.nanmedian(col_no_surface[:,column]) * 0.7
            col_avg.append(avg)
            sub_col.append(avg)
            sub_index.append(column)
    # col_avg = np.where(np.isnan(col_avg), np.nanmean(col_avg), col_avg)
    
    #run a cubic spline interpolation on the data
    if len(sub_index) != len(col_avg):
        interp = interp1d(sub_index, sub_col, kind = "linear")
        col_avg_pchip = interp(np.arange(len(col_avg)))
    else:
        col_avg_pchip = col_avg
    destriped = cube_band - col_avg_pchip
    now_time = time.time(); dicts["destripe"] = now_time - start_time; start_time = now_time
    cub = '_'.join(cube.fname.split("_")[0:2])
    if not os.path.exists(join_strings(SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["dev_figures_sub_path"], "polar_cache/",cub)):
        os.makedirs(join_strings(SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["dev_figures_sub_path"], "polar_cache/",cub))
    save_path = join_strings(SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["dev_figures_sub_path"], "polar_cache/", cub, str(band) + "_" + str(cube.wvlns[band-1]) + ".png")
    if not os.path.exists(save_path) or SETTINGS["processing"]["clear_cache"]:
        band_wave = str(band) + "_" + str(cube.wvlns[band-1])
        
        fig, axs = plt.subplots(2,3)
        axs[0,0].set_title("original surface calc")
        axs[0,1].set_title("cv2 surface calc")
        axs[0,2].set_title("plot of destripe")
        axs[1,0].set_title(cub + " " + band_wave)
        axs[1,1].set_title("gaussian blur")
        axs[1,2].set_title("destriped image")
        
        rgb_image = np.zeros((surface.shape[0], surface.shape[1], 3), dtype=np.uint8)
        rgb_image[:, :, 2] = np.where(surface, img, 0)
        rgb_image[:, :, 0] = np.where(surface, 0, img)
        rgb_image[:, :, 1] = 50

        axs[0,0].imshow(rgb_image)
        rgb_image = np.zeros((surface.shape[0], surface.shape[1], 3), dtype= np.uint8)
        rgb_image[:, :, :] = 50
        rgb_image[:, :, 2] = np.where(actual_surface, img, 0)
        rgb_image[:, :, 0] = np.where(actual_surface, 0, img)
        axs[0,1].imshow(rgb_image)
        
        
        axs[1,1].imshow(cv2.GaussianBlur(img, (5, 5), 0))

        axs[0,2].plot(np.arange(len(col_avg)), col_avg, label = "original")
        axs[0,2].plot(np.arange(len(col_avg)), col_avg_pchip, label = "pchip")
        axs[0,2].legend()
        
        axs[1,0].imshow(cube_band, cmap = "gray")
        axs[1,2].imshow(destriped, cmap = "gray")
        fig.tight_layout()
        now_time = time.time(); dicts["fig_gen"] = now_time - start_time; start_time = now_time
        plt.savefig(join_strings(SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["dev_figures_sub_path"], "polar_cache/", cub, band_wave + ".png"), dpi = 300)
        # plt.pause(0.1)
        plt.close()
    now_time = time.time(); dicts["save_fig"] = now_time - start_time; start_time = now_time
    return destriped
# flyby directory location
# name of csv containing all data (inside flyby folder)
# name of flyby image data (inside flyby folder)
# save loation (default to flyby data)

# surface_windows = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True]
# surface_windows = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True,
#                    True, True, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True]
# ir_surface_windows = [99, 100, 106, 107, 108, 109, 119, 120, 121, 122, 135, 136, 137, 138, 139, 140, 141, 142, 163, 164,
#                       165, 166, 167, 206, 207, 210, 211, 212, 213, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352]


class polar_profile:
    def __init__(self, north_orientation, distance_array: np.ndarray = None, cube_center: list = None, figures: bool = None,):
        self = self
        self.figures = {
            "original_image": False,
            "emission_heatmap": False,
            "emission_heatmap_overlay": False,
            "incidence_heatmap": False,
            "incidence_heatmap_overlay": False,
            "distance_from_center": False,
            "show_slant": False,
            "plot_polar": False,
            "full_cube_slant": False,
            "persist_figures": True,
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

    def get_lowest(self, arr):
        flattened_array = arr.flatten()
        lowest_indexes = np.argpartition(flattened_array, 5)[:4]
        lowest_indexes_2d = np.unravel_index(lowest_indexes, arr.shape)

        lowest_index = np.argmin(arr)
        lowest_index_2d = np.unravel_index(lowest_index, arr.shape)

        center = [lowest_index_2d, np.mean(lowest_indexes_2d, axis=1)]
        center = np.mean(center, axis=0)
        center_point = [int(np.round(center[0])), int(np.round(center[1]))]
        return center, center_point, lowest_index_2d, lowest_indexes_2d

    def calculate_location_of_lowest_inc(self, cube):
        lowest_inc, lowest_inc_point, lowest_index_2d, lowest_indexes_2d = self.get_lowest(cube.inc)
        if self.figures["incidence_heatmap"]:
            fig = plt.figure(self.figure_keys["incidence_heatmap"])
            plt.imshow(cube.inc)
            plt.scatter(lowest_indexes_2d[1], lowest_indexes_2d[0], s=4)
            plt.scatter(lowest_index_2d[1], lowest_index_2d[0], s=4)
            plt.scatter(lowest_inc[1], lowest_inc[0], c="r")
            self.show()
        if self.figures["incidence_heatmap_overlay"]:
            fig = plt.figure(self.figure_keys["incidence_heatmap_overlay"])
            plt.imshow(cube[int(cube.bands[68])])
            plt.scatter(lowest_indexes_2d[1], lowest_indexes_2d[0], s=4)
            plt.scatter(lowest_index_2d[1], lowest_index_2d[0], s=4)
            plt.scatter(lowest_inc[1], lowest_inc[0], c="r")
            self.show()
            self.saved_figures["incidence_heatmap_overlay"] = fig
        return lowest_inc

    def find_center_of_cube(self, cube):

        center, center_point, lowest_index_2d, lowest_indexes_2d = self.get_lowest(
            cube.eme)
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

    def slanted_cross_section(self, cube, cube_band, cube_upscaled, degree):
        """
            Get the cross section of the cube at a given angle relative to the center point (top is 0 degrees, going clockwise)
            We return the distance from the center point, the emission angle, the brightness, and the pixel indices of the cross section

            returns:
                ret = {
                    "pixel_indices" : [[x1,y1],[x2,y2],...], : pixel indices of the cross section
                    "pixel_distances" : [d1,d2,...], : distance from center point
                    "emission_angles" : [e1,e2,...], : emission angle of each pixel
                    "brightness_values" : [b1,b2,...], : brightness value of each pixel
                    "meta": {
                        "actual_angle" : degree, : the angle that was passed in
                        "angle_rad" : angle_rad, : the angle in radians
                        "starting_x_y_image" : [start_x,start_y], : the starting x and y coordinates of the line in the image
                        "ending_x_y_image" : [endpoint_x, endpoint_y] : the ending x and y coordinates of the line in the image,
                        "processing" : {
                            "sorted" : False, : whether or not the data has been sorted
                            "filtered" : False, : whether or not the data has been filtered
                            "smoothed" : False, : whether or not the data has been smoothed
                            "interpolated" : False, : whether or not the data has been interpolated
                    }
                }
        """
        scalar = cube_upscaled.shape[0] / cube_band.shape[0]
        angle_rad = np.radians(degree)
        center_point = np.around(self.cube_center)
        center_point = center_point * scalar + (scalar - 1) / 2
        start_x = center_point[1]
        start_y = center_point[0]
        dist = np.sqrt(
            (cube_upscaled.shape[0] - center_point[0]) ** 2
            + (cube_upscaled.shape[1] - center_point[1]) ** 2
        )
        endpoint_x = start_x + np.sin(angle_rad) * dist
        endpoint_y = start_y - np.cos(angle_rad) * dist
        mask = np.zeros(cube_upscaled.shape)
        cv2.line(
            mask, (int(start_x), int(start_y)), (int(
                endpoint_x), int(endpoint_y)), color=1, thickness=2)
        upscaled_indices = np.where(mask == 1)
        upscaled_indices = np.array(upscaled_indices).T

        # VISUALIZATION
        # plt.imshow(np.where(mask == 1, 0, cube_upscaled))
        # plt.show()

        line_indices = np.around(upscaled_indices / scalar, 0)
        # ensure that none are out of bounds
        line_indices = [[np.max((np.min((x,  cube_band.shape[0] - 1)), 0)), np.max(
            (np.min((y, cube_band.shape[1] - 1)), 0))] for x, y in line_indices]

        # remove repeats
        line_indices = np.unique(line_indices, axis=0).astype(int)

        # VISUALIZATIOn
        # mask = np.zeros(cube_band.shape)
        # mask[line_indices[:, 0], line_indices[:, 1]] = 1
        # plt.imshow(np.where(mask == 1, 0, cube_band))
        # plt.show()

        if self.figures["show_slant"]:
            fig = plt.figure(self.figure_keys["show_slant"])
            plt.scatter(line_indices[:, 1], line_indices[:, 0], marker="s")
            img = cube_band.copy()
            img[line_indices[:, 0], line_indices[:, 1]] = 0
            plt.imshow(img)
            self.show()
            self.saved_figures["show_slant_" + str(degree)] = fig
        try:
            distances = [self.distance_array[x, y] for x, y in line_indices]
            distances = np.array(distances)
        except:
            print(line_indices)
        brightness_values = [cube_band[x, y] for x, y in line_indices]
        # sort values by distance
        emission_angles = [cube.eme[x, y] for x, y in line_indices]

        if self.figures["plot_polar"]:
            pairs = list(zip(distances, brightness_values,
                         line_indices, emission_angles))

            # Sort the pairs based on distances
            sorted_pairs = sorted(pairs, key=lambda x: x[0])

            # Unpack the sorted pairs into separate arrays
            sorted_distances, sorted_brightness_values, pixel_indices, emission_angles = zip(
                *sorted_pairs)
            fig = plt.figure(self.figure_keys["plot_polar"])
            plt.plot(sorted_distances, sorted_brightness_values)
            self.show()
            self.saved_figures["plot_polar_" + str(degree)] = fig

        ret = {"pixel_indices": line_indices, "pixel_distances": distances, "emission_angles": emission_angles, "brightness_values": brightness_values, "meta": {"actual_angle": degree, "angle_rad": angle_rad,
                                                                                                                                                                 "starting_x_y_image": [start_x, start_y], "ending_x_y_image": [endpoint_x, endpoint_y], "processing": {"sorted": False, "filtered": False, "smoothed": False, "interpolated": False}}}
        return ret

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
        destriped_band
    ):
        self.cube_name = cube.img_id
        slants = [0, 30, 45, 60, 90, 120, 135, 150,
                  180, 210, 225, 240, 270, 300, 315, 330]

        degrees = self.north_orientation + np.array(slants)
        cmap = matplotlib.cm.get_cmap("rainbow")
        fits = {}

        # cube_band = cube[band]
        cube_band = destriped_band
        if band < 97:
            destrip = destripe_VIMS_V(cube,band)
        else:
            destrip = cube[band]
        if not np.all(cube_band == destrip):
            raise ValueError("BAD IMAGES")
        scalar = 5  # needs to be odd number
        cube_upscaled = cv2.resize(
            cube_band, (cube_band.shape[1] * scalar, cube_band.shape[0] * scalar))
        for index, degree in enumerate(degrees):
            transect = self.slanted_cross_section(
                cube, cube_band, cube_upscaled, degree)
            fits[slants[index]] = transect
            #transect testing
            slant_b = np.array([cube_band[pixel_index[0], pixel_index[1]]
                               for pixel_index in transect["pixel_indices"]])
            if not np.all(slant_b == transect["brightness_values"]):
                raise ValueError("Brightness values not equal between transect and equal test")

            slant_b = np.array([destrip[pixel_index[0], pixel_index[1]]
                               for pixel_index in transect["pixel_indices"]])            
            if not np.all(slant_b == transect["brightness_values"]):
                raise ValueError("Brightness values not equal between destripe recreation")
        if self.figures["full_cube_slant"]:
            for index, (pixel_distance, emission_angles, brightness, pixel_indices,) in enumerate(fits.values()):
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
        return fits


class complete_cube:
    def __init__(self, cube_name: str):
        # if data_save_folder_name is None:
        #     save_location = join_strings(
        #         parent_directory, cube_subdirectory, "analysis/limb/")
        # else:
        #     save_location = join_strings(
        #         parent_directory, cube_subdirectory, data_save_folder_name)
        self.cube_name = cube_name
        # self.result_data_base_loc = save_location
        # self.parent_directory = parent_directory
        # self.cwd = join_strings(self.parent_directory, cube_subdirectory)

    def convert_cube_to_dict(self, cube):
        ret = {
            "alt": cube.alt,
            "azi": cube.azi,
            "eme": cube.eme,
            "inc": cube.inc,
            "lat": cube.lat,
            "lon": cube.lon,
            "w": cube.w,
            "flyby": cube.flyby,
            "ground": cube.ground,
            "limb": cube.limb,
            "time": cube.time,
            "bands": [destripe_VIMS_V(cube, int(band), save = True) if int(band) <= 96 else cube[int(band)] for band in cube.bands],
            "original_bands": [cube[int(band)] for band in cube.bands]
        }

        return ret

    def inc_sampling(self, cube, center_point, rot_angle):
        objectx = polar_profile(0)
        inc_location = objectx.calculate_location_of_lowest_inc(cube)

        inc_rot = np.degrees(np.arctan2(
            inc_location[1] - center_point[1], center_point[0] - inc_location[0]))

        if inc_rot < 0:
            inc_rot += 360
        inc_rot_geo = inc_rot - rot_angle
        if inc_rot_geo > 360:
            inc_rot_geo -= 360

        # plt.plot([center_point[1], center_point[1] + np.sin(np.radians(rot_angle)) * 20],
        #             [center_point[0], center_point[0] - np.cos(np.radians(rot_angle)) * 20], c="r", label = "North")
        # plt.plot([center_point[1], center_point[1] - np.sin(np.radians(rot_angle)) * 20],
        #         [center_point[0], center_point[0] + np.cos(np.radians(rot_angle)) * 20], c="b")
        # plt.imshow(cube.inc)
        # plt.scatter(inc_location[1], inc_location[0], c="r", label = "Lowest Incidence")
        # plt.plot([center_point[1], center_point[1] + np.sin(np.radians(inc_rot)) * 15], [center_point[0], center_point[0] - np.cos(np.radians(inc_rot)) * 15], c ="g")
        # plt.scatter(center_point[1], center_point[0], c="b", label = "Center of Cube")
        # plt.title(str(inc_rot_geo))
        # plt.legend()
        # plt.show()
        return inc_rot_geo, inc_location

    def find_north_and_south(self, cube):
        objectx = polar_profile(0)
        distance_array, center_of_cube, center_point = objectx.distance_from_center_map(
            cube)

        center_of_cube, center_point = objectx.find_center_of_cube(cube)

        angles = []
        brightnesses = []
        for y in range(cube.lat.shape[0]):
            for x in range(cube.lat.shape[1]):
                # if cube.ground[y, x] == True:
                #     continue
                brightnesses.append(cube.lat[y, x]) # brightness value
                angles.append(
                    int(np.degrees(np.arctan2(x - center_point[1], center_point[0] - y)))) # angle relative to center
        
        actual_angles = np.array(sorted(set(angles))) #angle integers
        br_per_int_angle = [([brightnesses[index] for index, a in enumerate(
            angles) if a == angle]) for angle in actual_angles] #brightness values for each angle
        for index, br in enumerate(br_per_int_angle):
            br_per_int_angle[index] = br[np.argmax(np.abs(br))]

        selected_minimums = []
        selected_maximums = []

        initial_amplitude = 90  # Since values range from -90 to 90
        initial_frequency = 1  # One cycle per 360 degrees
        initial_phase = 0  # Initial phase
        initial_offset = 0  # Initial offset

        # Adjust bounds as needed
        param_bounds = ([0, 0, -360, -180], [100, 10, 360, 180])

        initial_guess = [initial_amplitude,
                        initial_frequency, initial_phase, initial_offset]
        fitted = False
        # try:
        #     # Use curve_fit to fit the triangle wave function to your data
        #     params, covariance = curve_fit(
        #         triangle_wave, actual_angles, gaussian_filter(br_per_int_angle,sigma = 3), p0=initial_guess, bounds=param_bounds)
        #     lin_angles = np.linspace(0, 360, 3600)
            
        #     diffs = triangle_wave(actual_angles, *params) - br_per_int_angle
        #     if np.std(diffs) > 0.15*np.std(br_per_int_angle):
        #         print(np.std(diffs) , np.std(br_per_int_angle))
        #         raise ValueError
        
        #     values_from_fit = triangle_wave(lin_angles, *params)
        #     sorted_values = np.argsort(values_from_fit)
        #     min_angle = lin_angles[sorted_values[0]]
        #     max_angle = lin_angles[sorted_values[-1]]
        #     fitted = True
        # except:
            # print("failed to fit triangle wave")
            # plt.scatter(angles, brightnesses)
            # plt.scatter(actual_angles, br)
            # plt.scatter(actual_angles, gaussian_filter(br, sigma=3))
            # plt.pause(3)
        sorted_values = np.argsort(br_per_int_angle)
        actual_angles = np.array(actual_angles)

        selected_minimums = actual_angles[sorted_values[0:20]]
        if np.std(selected_minimums) > 10:
            selected_minimums = [m+360 if m < 0 else m for m in selected_minimums]
                        
        min_angle =np.mean(selected_minimums)
        
        selected_maximums = actual_angles[sorted_values[-20::]]
        if np.std(selected_maximums) > 10:
            selected_maximums = [m+360 if m < 0 else m for m in selected_maximums]
        max_angle =np.mean(selected_maximums)
        
        std_thresh = 10
        weightage_north = np.count_nonzero(np.array(angles) > 0) / len(angles)
        weightage_south = np.count_nonzero(np.array(angles) < 0) / len(angles)
        
        
        if min_angle < 0:
            min_angle += 360
        if max_angle < 0:
            max_angle += 360
        calculated_rots = np.array([min_angle, max_angle])
        # if np.std(selected_minimums) > std_thresh and np.std(selected_maximums) > std_thresh:
        #     raise ValueError

        if min_angle > max_angle:
            rot_angle = min_angle * weightage_south + max_angle * weightage_north - 90
        else:
            rot_angle = min_angle * weightage_south + max_angle * weightage_north + 90
        
        if rot_angle > 180:
            rot_angle -= 360
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].scatter(angles, brightnesses, label = "Original")
        axs[0].scatter(actual_angles, br_per_int_angle, label = "Average")
        axs[0].scatter(actual_angles, gaussian_filter(br_per_int_angle, sigma=3), label = "Gaussian Filter")
        axs[0].vlines(rot_angle, np.min(br_per_int_angle), np.max(br_per_int_angle), label = "North", color = (0,0,0), linewidth = 2)
        axs[0].vlines(rot_angle - 180, np.min(br_per_int_angle), np.max(br_per_int_angle), label = "South", color = (0,0,0), linestyle = "--", linewidth = 2)

        axs[0].legend()
        axs[1].imshow(cube.lat)
        try:
            axs[2].imshow(cube[69])
            plt.title("VIS")
        except:
            axs[2].imshow(cube[118])
            plt.title("IR")
        shs= cube.lat.shape[0]
        axs[1].plot([center_point[1], center_point[1] + np.sin(np.radians(rot_angle)) * shs],
                    [center_point[0], center_point[0] - np.cos(np.radians(rot_angle)) * shs], c="r")
        axs[2].plot([center_point[1], center_point[1] + np.sin(np.radians(rot_angle)) * shs],
                    [center_point[0], center_point[0] - np.cos(np.radians(rot_angle)) * shs], c="r")
        axs[1].plot([center_point[1], center_point[1] - np.sin(np.radians(rot_angle)) * shs],
                    [center_point[0], center_point[0] + np.cos(np.radians(rot_angle)) * shs], c=(0.5,0,0))
        axs[2].plot([center_point[1], center_point[1] - np.sin(np.radians(rot_angle)) * shs],
                    [center_point[0], center_point[0] + np.cos(np.radians(rot_angle)) * shs], c=(0.5,0,0))
        if not os.path.exists(join_strings(SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["dev_figures_sub_path"], "polar_cache/north/")):
            os.makedirs(join_strings(SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["dev_figures_sub_path"], "polar_cache/north/"))
        # plt.show()
        plt.savefig(join_strings(SETTINGS["paths"]["parent_figures_path"], SETTINGS["paths"]["dev_figures_sub_path"], "polar_cache/north/" + cube.fname.split(".")[0]+".png"), dpi = 300)
        plt.close()
        return rot_angle, distance_array, center_of_cube, center_point

    def run_analysis(
        self,
        cube,
        north_orientation,
        distance_array,
        center_of_cube,
        destriped_band,
        band,
        band_index,
        key,
    ):
        start = time.time()
        analysis = polar_profile(north_orientation=north_orientation,
                                 distance_array=distance_array, cube_center=center_of_cube)
        data = analysis.complete_image_analysis_from_cube(
            cube, int(band), band_index=band_index, destriped_band = destriped_band)

        end = time.time()
        total_time = end - self.start_time
        percentage = (band - self.starting_band + 1) / \
            (self.total_bands - self.starting_band)

        print(key, "took", np.round(end - start,2), "seconds.", "expected time left:",
              np.round(total_time / percentage - total_time,2), "seconds                          ", end="\r")
        return data

    def analyze_dataset(self, cube_root: str, cube: str = None, force=False):
        if cube is None:
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
        (north_orientation, distance_array, center_of_cube,
         center_point) = self.find_north_and_south(self.cube_vis)
        incident_angle, incident_location = self.inc_sampling(
            self.cube_vis, center_point, north_orientation)
        datas = {}
        datas["meta"] = {
            "cube_vis":  self.convert_cube_to_dict(self.cube_vis),
            "cube_ir": self.convert_cube_to_dict(self.cube_ir),
            "lowest_inc_vis": incident_angle,
            "lowest_inc_location_vis": incident_location,
            "north_orientation_vis": north_orientation,
            "center_of_cube_vis": center_of_cube,
        }
        print() #clears any saves
        for band in self.cube_vis.bands:
            band_index = int(band) - 1
            # if surface_windows[band_index]:
            #     continue
            key = str(self.cube_vis.w[band_index]) + "µm_" + str(int(band))
            if self.starting_band == -1:
                self.starting_band = int(band)
            data = self.run_analysis(self.cube_vis, north_orientation=north_orientation, distance_array=distance_array,
                                     center_of_cube=center_of_cube, destriped_band = datas["meta"]["cube_vis"]["bands"][band_index], band=band, band_index=band_index, key=key)
            datas[key] = data

        (north_orientation, distance_array, center_of_cube,
         center_point) = self.find_north_and_south(self.cube_ir)
        incident_angle, incident_location = self.inc_sampling(
            self.cube_vis, center_point, north_orientation)
        print("\nVIS done") #clears any saves
        for band in self.cube_ir.bands:
            band_index = int(band) - 97
            # if int(band) in ir_surface_windows:
            #     continue
            key = str(self.cube_ir.w[band_index]) + "µm_" + str(int(band))
            if self.starting_band == -1:
                self.starting_band = int(band)
            data = self.run_analysis(self.cube_ir, north_orientation=north_orientation, distance_array=distance_array,
                                     center_of_cube=center_of_cube, destriped_band = datas["meta"]["cube_ir"]["bands"][band_index], band=band, band_index=band_index, key=key)
            datas[key] = data
        print("\nIR done") #clears any saves
        datas["meta"]["lowest_inc_ir"] = incident_angle
        datas["meta"]["north_orientation_ir"] = north_orientation
        datas["meta"]["center_of_cube_ir"] = center_of_cube
        datas["meta"]["lowest_inc_location_ir"] = incident_location

        #             data.append(
        #     {
        #         "flyby": cube.flyby.name,
        #         "cube": cube.img_id,
        #         "band": cube.w[band_index],
        #         "center_of_cube": center_of_cube,
        #         "north_orientation": north_orientation,
        #     }
        # )

        return datas


class analyze_complete_dataset:
    def __init__(self, cubes_location: str = None) -> None:
        if cubes_location is None:
            cubes_location = join_strings(
                SETTINGS["paths"]["parent_data_path"],SETTINGS["paths"]["cube_sub_path"],
            )
        self.cubes_location = cubes_location
        folders = os.listdir(self.cubes_location)
        self.cubes = [
            folder
            for folder in folders
            if folder.startswith("C") and folder[-2] == ("_")
        ]
        self.cubes.sort()
        
    def time_calculations(self, end, start, index, total):
        index +=1
        delta = np.around(end - start,2)
        total_time = np.around(total/index * delta,2)
        time_left = np.around(total_time - delta,2)
        print(str(index) + "/" + str(total), "| time spent:", delta, "| time left:", time_left,"| total time:", total_time , "\n")

    def run_func(self, cub, fit_cube, end, start, index, total, force_write):
        image = complete_cube(cub)
        datas = image.analyze_dataset(cube_root=join_strings(self.cubes_location, cub), cube=cub, force=force_write)
        check_if_exists_or_write(cub + ".pkl", base=fit_cube, prefix="", save=True, data=datas, force_write=force_write, verbose=True)
        self.time_calculations(end, start, index, total)
        return (cub, datas)  # Return a tuple with the cube identifier and the data


    def complete_dataset_analysis(self, multi_process : Union[bool, int] = True):
        self.all_data = {}
        force_write = (SETTINGS["processing"]["clear_cache"]
                       or SETTINGS["processing"]["redo_polar_profile_calculations"])
        appended_data = False
        
        
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
            
        fit_cube = join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["analysis_sub_path"])
        if all([cub + ".pkl" in os.listdir(fit_cube) for cub in self.cubes]) and os.path.exists(join_strings(fit_cube, get_cumulative_filename("analysis_sub_path"))) and  not force_write:
            print("Polar profiles already calculated")
            return
        
        self.all_start_time = time.time()
        leng = len(self.cubes)
        for index, cub in enumerate(self.cubes):

            if not os.path.exists(fit_cube):
                os.makedirs(fit_cube)
            if cub + ".pkl" in os.listdir(fit_cube) and not force_write:
                self.all_data[cub] = check_if_exists_or_write(
                    cub + ".pkl", base=fit_cube, prefix="", save=False, data=None, verbose=True)
                continue
                
            elif not force_write:
                appended_data = True
            if multi_process:
                args.append([cub, fit_cube, time.time(), self.all_start_time, index, leng, force_write])
            else:
                cub, datas = self.run_func(cub, fit_cube, time.time(), self.all_start_time, index, leng, force_write=force_write)
                self.all_data[cub] = datas
        
        if multi_process:
            with multiprocessing.Pool(processes=multi_process_core_count) as pool:
                results = pool.starmap(self.run_func, args)
            # Aggregate the results into self.all_data
            for cub, data in results:
                self.all_data[cub] = data
            
            check_if_exists_or_write(get_cumulative_filename("analysis_sub_path"),join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["analysis_sub_path"]), save=True, data=self.all_data, force_write=force_write or appended_data or not os.path.exists(join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["analysis_sub_path"],get_cumulative_filename("analysis_sub_path"))), verbose=True)
