import os
import os.path as path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from PIL import Image, ImageStat
from scipy.optimize import curve_fit
import time
import math
from sklearn.metrics import r2_score
import PIL
import pyvims
from .equirectangular import equi_cube
from get_settings import join_strings, check_if_exists_or_write, SETTINGS
from scipy.ndimage import gaussian_filter
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
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
class nsa_analysis:   
    def __init__(self, cube_vis, cube_ir, cube_name, usable_bands: list = None):
        self.cube_vis = cube_vis
        self.cube_ir = cube_ir
        self.cube_name = cube_name
        if usable_bands is not None:
            self.usable_bands = usable_bands
        else:
            self.usable_bands = SETTINGS["figure_generation"]["nsa_bands"]
        #create all class paths,directories, and variables
        # self.createDirectories(directory, datasetName)
        # self.createLists(purpose, datasetName)
        # self.analysisPrep(shiftDegree)
        #gets purpose condition and executes individual operations based on purpose
        self.figures = SETTINGS["figure_generation"]["nsa_figs"]
        self.figure_keys = {key: index for index, (key, value) in enumerate(self.figures.items())}

    def show(self, force = False, duration : float = None):
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
            return
        else:
            plt.pause(2)
            plt.clf()
            plt.close()

    def bound_cube_geo(self, cube):
        #compare new image to old image (lats and lons)
        #get the max and min ground lat and lon to get the range of values to iterate over (once per cube(vis or ir)
        
        return [[np.min(cube.ground_lat), np.max(cube.ground_lat)],[np.min(cube.ground_lon), np.max(cube.ground_lon)]]
    def pixel_to_geo(self, pixel, geo_list): #convert pixel to lat or pixel to lon
        return geo_list[int(np.around(pixel))]
    

    def shift_image(self, image, shift, crop : bool = True):
        subtraction = (np.insert(image, 0, np.array(shift*2*[[0]*image.shape[1]]), axis=0) - np.concatenate((image, shift*2*[[0]*image.shape[1]])))
        hc_band = subtraction[2*shift:-2*shift]
        #add nan rows to top and bottom of image
        nan_rows = [False for i in range(shift)] + [True for i in range(hc_band.shape[0])] + [False for i in range(shift)]
        hc_band = np.insert(hc_band, 0, shift*[[np.nan]*hc_band.shape[1]], axis=0)
        hc_band = np.insert(hc_band, hc_band.shape[0], shift*[[np.nan]*hc_band.shape[1]], axis=0)
        
        if crop:
            imager = image.filled(fill_value=np.nan)
            nan_columns = np.isnan(imager).sum(axis=0) > 0.15 * image.shape[0]
            hc_band = hc_band[:, ~nan_columns]
            return hc_band, nan_rows, nan_columns
        
        return hc_band, nan_rows
    
    def equirectangular(self, cube, band : int):
        # take image and apply cylindrical projection
        # after cylindrical projection, remove extraneous longitude data
        proj = pyvims.projections.equirectangular.equi_cube(cube,int(band),3)
        #equirectangular projection
        # img, (x, y), extent, cnt = equi_cube(c, band, n=512, interp="cubic")
        if self.figures["show_projection"]:
            plt.imshow(proj[0], origin="upper", cmap="gray")
            self.show()
        # plt.plot(glon, glat, 'k.', ms=1)
        return proj
    
    def n_fit(self, x, *args):
        return np.polyval(args, x)
    def n_derivative(self, *args):
        return [args[len(args)-1 - i]*i for i in range(len(args)-1, 0, -1)]
    def n_derivative_plot(self, x, *args):
        fit = self.n_derivative(*args)
        return self.n_fit(x, *fit)

    def curve_fitting(self, x, y, order, weightage = None):
        popt, pcov = np.polyfit(x, y, order, w = weightage, cov = True)
        # if weightage is None:
        #     popt, pcov = curve_fit(self.octic_fit, x, y)
        # else:
        #     popt, pcov = curve_fit(self.octic_fit, x, y, sigma = weightage, absolute_sigma = True)
        # plot = self.sextic_fit(x, *popt)
        # derivative = self.sextic_derivative(*popt)
        # derivative_plot = self.sextic_derivative_plot(x, *popt)
        
        plot = self.n_fit(x, *popt)
        derivative = self.n_derivative(*popt)
        derivative_plot = self.n_derivative_plot(x, *popt)
        
        r2 = r2_score(y, plot)
        return popt, pcov, plot, derivative, derivative_plot, r2
    def polyfit(self, x, y, degree): #alternate fit for polynomials
        results = {}
        coeffs = np.polyfit(x, y, degree)
        p = np.poly1d(coeffs)
        #calculate r-squared
        yhat = p(x)
        ybar = np.sum(y)/len(y)
        ssreg = np.sum((yhat-ybar)**2)
        sstot = np.sum((y - ybar)**2)
        return ssreg / sstot
    def gaussian(self, x, amplitude, mean, stddev): #gaussian fit
        return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)
    def poly6(self, x, g, h, i, j, a, b, c): #sextic function
        return g*x**6+h*x**5+i*x**4+j*x**3+a*x**2+b*x+c
    def poly6Prime(self, x, g, h, i, j, a, b, c): #derivative of sextic function
        return 6*g*x**5+5*h*x**4+4*i*x**3+3*j*x**2+2*a*x+b
    def poly6Derivative(self, g, h, i, j, a, b, c): #derivative of sextic function coefficents
        return [6*g,5*h,4*i,3*j,2*a,b]
    def running_mean(self, x, N): #window avearage
        return np.convolve(x, np.ones(N)/N, mode='valid')
    
    def getDataControl(self): #iteration over images within flyby
        for self.iter in range(len(self.allFiles)):
            self.a = time.time()
            self.currentFile = self.allFiles[self.iter]
            self.iterations.append(self.iter)
            self.dataAnalysis()
            print("dataset", self.flyby, "     image", "%02d" % (self.iter),"     boundary",format(self.NSA[self.iter],'.15f') ,"      deviation", format(self.deviation[self.iter],'.15f'), "        N/S",format(self.NS[self.iter],'.10f') )
        #write data
        if "write" in self.purpose[1]:
            #create folder and file
            self.createFolder()
            self.createFile()
            try:
                datasetAverage = self.averages([self.NSA, np.std(self.NSA), self.NS])
                self.NSA.append(datasetAverage[0]); self.deviation.append(datasetAverage[1]); self.NS.append(datasetAverage[2]); self.iterations.append(self.flyby)
                self.NSA.insert(0, "NSA"); self.deviation.insert(0, "Deviation"); self.NS.insert(0,"N/S"); self.iterations.insert(0, "File Number")
                self.fileWrite(self.NSA)
                self.fileWrite(self.deviation)
                self.fileWrite(self.NS)
                self.fileWrite(self.iterations)
            finally:
                self.fileClose()
    def visualizeBrightnessDifferenceSamplingArea(self, im, x, nsaLat):
        if type(self.leftCrop) is list:
            im[x[0]:x[1],self.rightCrop[0]:self.leftCrop[0]] *= 2
            im[(x[1]+1):x[2],self.rightCrop[0]:self.leftCrop[0]] *=  0.5
            im[x[0]:x[1],self.rightCrop[1]:self.leftCrop[1]] *= 2
            im[(x[1]+1):x[2],self.rightCrop[1]:self.leftCrop[1]] *=  0.5
        else:
            im[x[0]:x[1],self.leftCrop:self.rightCrop] *= 2
            im[(x[1]+1):x[2],self.leftCrop:self.rightCrop] *=  0.5
        plt.imshow(im, cmap = 'Greys')
        plt.show()
    def brightnessDifference(self, im, nsaLat):
        splitY = min(range(len(self.lat[:, 0])), key=lambda x:abs(self.lat[x,0]-nsaLat))
        horizontalSample= 4
        verticalSample = 30
        northSplit = min(range(len(self.lat[:, 0])), key=lambda x:abs(self.lat[x,0]-verticalSample-nsaLat))
        southSplit = min(range(len(self.lat[:, 0])), key=lambda x:abs(self.lat[x,0]+verticalSample-nsaLat))
        
        if type(self.leftCrop) is list:
            north = im[northSplit:splitY,self.rightCrop[0]:self.leftCrop[0]]
            south = im[(splitY+1):southSplit,self.rightCrop[0]:self.leftCrop[0]]
            north = np.concatenate((north, im[northSplit:splitY,self.rightCrop[1]:self.leftCrop[1]]), axis = 1)
            south = np.concatenate((south,im[(splitY+1):southSplit,self.rightCrop[1]:self.leftCrop[1]]), axis = 1)
        else:
            if self.leftCrop > self.rightCrop:
                north = im[northSplit:splitY,self.rightCrop:self.leftCrop]
                south = im[(splitY+1):southSplit,self.rightCrop:self.leftCrop]
            else:
                north = im[northSplit:splitY,self.leftCrop:self.rightCrop]
                south = im[(splitY+1):southSplit,self.leftCrop:self.rightCrop]
        northM = np.mean(north[north != 0.])
        southM = np.mean(south[south != 0.])
        return northM/southM
    def dataAnalysis(self):
        try: #open image arrays
            self.im = plt.imread(self.currentFile)[:,:,0]
        except:
            self.im = plt.imread(self.currentFile)[:,:]
        self.im = self.im.astype(np.int16)
        hc_band = np.empty((self.height, self.width), float)
        nsa_lats = []; nsa_lons = []; cols = []
        latRange = np.max(self.lat)-np.min(self.lat)
        latTicks = len(self.lat)/latRange
        #get latitude values of each pixel using CSV
        width = []
        subtraction = (np.insert(self.im, 0, np.array(self.num_of_nans*[[0]*self.width]), axis = 0) - np.concatenate((self.im,self.num_of_nans*[[0]*self.width])))
        hc_band = subtraction[int(np.round(self.num_of_nans/2, 0)):-1*int(np.round(self.num_of_nans/2, 0))]
        if True:
            self.createFolder(os.path.join(self.masterDirectory,self.flyby,"hc_band"))
            self.convert_array_to_image(hc_band, os.path.join(self.masterDirectory,self.flyby,"hc_band", (os.path.splitext(os.path.relpath(self.currentFile,os.path.join(self.masterDirectory,self.flyby,self.directoryList["flyby_image_directory"])))[0] +  "_band")))
        if_sh = hc_band[self.subset]
        lon_subset = []
        if type(self.leftCrop) is list:
            a =  sorted((self.rightCrop[0],self.leftCrop[0]))
            b =  sorted((self.rightCrop[1],self.leftCrop[1]))
            lon_subset = np.concatenate((range(*a), range(*b)))
        else:
            if self.leftCrop > self.rightCrop:
                lon_subset = range(self.rightCrop,self.leftCrop)
            else:
                lon_subset = range(self.leftCrop,self.rightCrop)
        for col in range(self.width):
            if col in lon_subset:
                columnHC = hc_band[:,col]
                if_sh = columnHC[self.subset] ## subset HC band b/t 30°S to 0°N 
                if np.min(if_sh) != np.max(if_sh):
                    try:
                        popt, _ = curve_fit(self.poly6, self.lat_sh, if_sh)#apply sextic regression to data
                        poptD = self.poly6Derivative(*popt)#get derivative of sextic regression
                        derivativeRoot = np.roots(poptD) #roots (Real and imaginary) of derivative function
                        realDerivativeRoots = derivativeRoot[np.isreal(derivativeRoot)] #remove extraneous soulutions (imaginary)
                        drIndex = min(range(len(realDerivativeRoots)), key=lambda x: abs(realDerivativeRoots[x]-self.goalNSA)) #find value closest to NSA
                        derivativeRoots = realDerivativeRoots[drIndex]
                        if abs(derivativeRoots.real-self.goalNSA) >= self.errorMargin:
                            width.append(False)
                        else: 
                            nsa_lats.append(derivativeRoots.real)
                            width.append(True)
                    except:
                        width.append(False)
            else:
                width.append(False)
        self.NSA_Analysis(nsa_lats, self.im,width)
        print(time.time() - self.a)
        #open image arrays
        try:
            self.im = plt.imread(self.currentFile)[:,:,0]
        except:
            self.im = plt.imread(self.currentFile)[:,:]
        self.im = self.im.astype(np.float32)
        self.hc_band = np.empty((self.height, self.width), float)
        #get latitude values of each pixel using CSV
        count = 0
        non_zero = np.array(np.any(self.im != 0,axis=1))
        image = self.im[non_zero, :]
        crop = self.im[non_zero, :]
        ab = self.lat[non_zero,0]
        Result = image[:,~np.any(crop == 0, axis = 0)]
        try:
            b = Result[int(len(Result)*0.25):int(len(Result)*0.75), int(len(Result[0])*0.25):int(len(Result[0])*0.75)]
        except:
            try:
                self.im = plt.imread(self.currentFile)[:,:,0]
            except:
                self.im = plt.imread(self.currentFile)[:,:]
            b = 0
        b = np.mean(b)*10
        a = np.mean(Result, axis = 1)
        ab = ab[abs(a)<abs(b)]
        a = a[abs(a)<abs(b)]
        return [ab, a]
    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
        #open image arrays
        if self.iter in self.purpose[1]:
            try:
                self.im = plt.imread(self.currentFile)[:,:,0]
            except:
                self.im = plt.imread(self.currentFile)[:,:]
            self.im = self.im.astype(np.float32)
            self.hc_band = np.empty((self.height, self.width), float)
            nsa_lats = []; nsa_lons = []; cols = []
            columns = []
            lon_shTilt = []
            for col in self.purpose[2]:
                x = self.columnAnalysis(col)
                if x != None:
                    nsa_lats.append(x)
                    lon_shTilt.append(self.lon[0,col])
                    columns.append(col)
                
            self.band.append([columns, lon_shTilt, self.smooth(nsa_lats,self.purpose[4])])

    def NSA_Analysis(self, im_nsa_lat,image, x):
        dev = np.std(im_nsa_lat) #standard deviation
        average = np.nanmean(im_nsa_lat) #standard average
        combo = 4
        movingAverageList = self.running_mean(im_nsa_lat, combo) #moving average
        # if "showAverage" in self.purpose[1]:
        #     plt.plot(range(len(movingAverageList)),movingAverageList)
        #     plt.show()
        movingAvg = np.mean(movingAverageList) #moving average
        diff = self.brightnessDifference(image, movingAvg) #difference between north and south
        self.NSA.append(movingAvg)
        self.deviation.append(dev)
        self.NS.append(diff)
    def columnAnalysis(self,column):
        subtraction = (np.insert(self.im[:,column], [0]*self.num_of_nans, self.nans) - np.concatenate((self.im[:,column], self.nans)))
        self.hc_band[:,column] = subtraction[int(self.num_of_nans/2):int(-self.num_of_nans/2)]
        #hc_band[crop[0]:crop[1],crop[2]:crop[3]]
        columnHC = self.hc_band[:,column]
        if_sh = columnHC[self.subset] ## subset HC band b/t 30°S to 0°N 
        #lat_sh = self.columnLat[self.subset]  ## subset HC band b/t 30°S to 0°N 
        self.lon_sh = self.lon[:,column][self.subset]  ## subset HC band b/t 30°S to 0°N 
        
        try:
            popt, _ = curve_fit(self.poly6, self.lat_sh, if_sh)#apply sextic regression to data
            poptD = self.poly6Derivative(*popt)#get derivative of sextic regression
            derivativeRoot = np.roots(poptD) #roots (Real and imaginary) of derivative function
            derivativeRoots = derivativeRoot[np.isreal(derivativeRoot)] #remove extraneous soulutions (imaginary)
            drIndex = min(range(len(derivativeRoots)), key=lambda x: abs(derivativeRoots[x]-self.goalNSA)) #find value closest to NSA
            derivativeRoots = derivativeRoots[drIndex] #get lat of derivative root
            derivativeRoots.real #append to nsa_lats
            return derivativeRoots.real
        except:
            return None
    def analyze_column(self, column):
        return column

    def compare_shifted_and_original(self, shifted_image, original_image):
        shifted_image= np.ma.masked_array(shifted_image, mask=original_image.mask)
        return shifted_image
    def find_zeros(self, f, interval):
        zeros = []
        x = np.linspace(interval[0], interval[1], 1000)
        y = f(x)
        for i in range(len(x) - 1):
            if np.sign(y[i]) != np.sign(y[i + 1]):
                zero = brentq(f, x[i], x[i + 1])
                zeros.append(zero)
        return zeros
    def weightage(self, projected_lat, idx):
        idx = (np.abs(projected_lat - idx)).argmin()

        weights = 1.0 / (1.0 + np.abs(np.arange(len(projected_lat)) - idx))**1
        return weights
    def check_if_good_val(self, projected_lat, col_data, relative_extrema,index_of_extrema, derivative):
        leng = len(col_data); filter_size = int(leng/10)
        proj_area = projected_lat[index_of_extrema - filter_size: index_of_extrema + filter_size]
        proj_col = col_data[index_of_extrema - filter_size: index_of_extrema + filter_size]
        sorted_values = np.argsort(proj_col)
        
        second_derivative = derivative(projected_lat[index_of_extrema+1]) - derivative(projected_lat[index_of_extrema-1])
        second_derivative /= projected_lat[index_of_extrema+1] - projected_lat[index_of_extrema-1]
        if second_derivative >= 0:
            xa = proj_col[sorted_values[10]]
            val = col_data[index_of_extrema]

            # if second derivative is positive, then it is a local min
            #if the 10th smallest value is the same as the value at the relative extrema, then it is a local minimum
            if proj_col[sorted_values[10]] > col_data[index_of_extrema]:
                ret = True
            else:
                ret = False
        else: 
            #you have a max here
            xa = proj_col[sorted_values[-20]]
            val = col_data[index_of_extrema]
            if proj_col[sorted_values[-20]] < col_data[index_of_extrema]:
                ret = True
            else:
                ret = False


        # plt.title(str(second_derivative) + " and " + str(ret))
        # plt.plot(proj_area, proj_col)
        # # plt.plot(proj_area, second_derivative(proj_area))
        # plt.plot(proj_area, derivative(proj_area))

        # plt.vlines(projected_lat[idx], ymin = np.min(proj_col), ymax = np.max(proj_col))
        # plt.show()
        return ret
        # return ret,
        # slope, shift = np.polyfit(proj_col, proj_area, 1)
        # relative_extrema =  col_data[idx]
    def find_index_of_closest(self, arr, idx):
        return (np.abs(arr - idx)).argmin()
    def evaluate_zeros(self, projected_lat, col_data, extra_guass_data, relative_extrema, derivative, original_guess = None):
        within_range = []; relative_extremas = []; ret_zero = None
        relative_indexes = {zero: self.find_index_of_closest(projected_lat, zero) for zero in relative_extrema}
        for zero in relative_extrema:
            if zero < -25 or zero > 25:
                continue
            elif self.check_if_good_val(projected_lat, extra_guass_data, zero, relative_indexes[zero], derivative):
                relative_extremas.append(zero)
            else:
                within_range.append(zero)
        #evaluate options
        if len(relative_extremas) == 0:
            return None
        elif len(relative_extremas) == 1:
            return relative_extremas[0], within_range
        
        if original_guess is not None:
            relative_extremas_close_to_guess = np.array([zero for zero in relative_extremas if zero > original_guess - 8 and zero < original_guess + 8])
            if len(relative_extremas_close_to_guess) == 1:
                return relative_extremas_close_to_guess[0], relative_extrema
            elif len(relative_extremas_close_to_guess) == 0:
                return None
            else:
                relative_extremas = relative_extremas_close_to_guess
                
        sorted_gauss = list(np.argsort( extra_guass_data[self.find_index_of_closest(projected_lat, -25): self.find_index_of_closest(projected_lat, 25)])  + self.find_index_of_closest(projected_lat, -25))
        indexes_sorted = [sorted_gauss.index(relative_indexes[val]) for val in relative_extremas]
        indexes_dist = [min([len(projected_lat) - index, index]) for index in indexes_sorted]
        index_to_ret = np.argmin(indexes_dist)
        return relative_extremas[index_to_ret], relative_extremas
        # ret_zero = ret_zeros[index_to_ret]

    def local_determination(self, projected_image, projected_lat, projected_lon, band, original_guess = None):
        proj_lat =projected_lat[:,0]
        # if original_guess is not None:
        #     weightage = self.weightage(proj_lat,original_guess)
        returns = []
        for column in range(0, projected_image.shape[1], 1):
            col_data = projected_image[:,column]
            lat_data = proj_lat
            mask = np.isfinite(col_data) & np.isfinite(lat_data)
            col_data = col_data[mask]
            lat_data = lat_data[mask]
            # if original_guess is not None:
            #     weight = weightage[mask]
            gauss_data = gaussian_filter(col_data,sigma=5)
            extra_gauss_data = simple_moving_average(gaussian_filter(col_data,sigma=10), 10)

            # fit_popt, fit_pcov, fit_plot, derivative, derivative_plot, r2 = self.curve_fitting(lat_data, super_gauss_data, 6, weight)
            # plt.plot(lat_data, gauss_data, color = (1,0,0), label = "gaussian")
            # plt.plot(lat_data, col_data, color = (0,1,0), label = "original")
            # plt.plot(lat_data, extra_gauss_data, color = (0,0,1), label = "extra gaussian")
            # plt.plot(lat_data, fit_plot, color = (0,0.5,0.5), label = "fit")
            pchip = PchipInterpolator(lat_data, extra_gauss_data)
            derivative = pchip.derivative()
            second_derivative = derivative.derivative()
            zeros = self.find_zeros(derivative, [np.min(lat_data), np.max(lat_data)])
            val = self.evaluate_zeros(lat_data, gauss_data, extra_gauss_data, zeros, derivative, original_guess)
            if val is None:
                continue
            else:
                ret_zero, ret_zeros = val
            if self.figures["zeros_evaluated"]:
                plt.plot(np.linspace(np.min(lat_data), np.max(lat_data),100),pchip(np.linspace(np.min(lat_data), np.max(lat_data),100)), )
                plt.vlines(zeros, ymin = np.min(col_data), ymax = np.max(col_data), color = (0,0,1), label = "detected zero")
                plt.vlines(ret_zeros, ymin = np.min(col_data), ymax = np.max(col_data), color = (0,1,1), label = "detected zero in range")
                plt.vlines(ret_zero, ymin = np.min(col_data), ymax = np.max(col_data), color = (0,1,0.5), label = "seleced zero")

                plt.legend()
                plt.show()
            returns.append(ret_zero)
        
        return returns, np.mean(returns)
        #     plt.plot(projected_lat, gaussian_filter(col_data,sigma=5), color = (colors, 0,0))

    
    def global_determination(self, projected_image, projected_lat, projected_lon, band):
        #provide projected image with as much filtering/processing as possible (just keep shape the same)
        projected_image = projected_image.flatten()
        projected_lat = projected_lat.flatten().astype(int)
        indexed_lats = {}
        for key in np.unique(projected_lat):
            values = projected_image[projected_lat == key]
            if np.ma.all(np.ma.getmask(values)):
                continue
            else:
                mean_value = np.nanmean(values)
            if not np.isnan(mean_value):
                indexed_lats[key] = mean_value

        # Convert the result to a NumPy array
        indexed_if = np.array(list(indexed_lats.values()))
        indexed_lats = np.array(list(indexed_lats.keys()))
        

        gauss = gaussian_filter(indexed_if, sigma = 8)
        pchip = PchipInterpolator(indexed_lats,  gauss)
        derivative = pchip.derivative()
        zeros = self.find_zeros(derivative, [np.min(indexed_lats), np.max(indexed_lats)])
        ret_zero = [zero for zero in zeros if zero > -30 and zero < 30]

        if self.figures["global_nsa_and_scatter"]:
            plt.scatter(projected_lat, projected_image, label = "original")
            plt.plot(indexed_lats, indexed_if, color = (1,0,0), label = "indexed")
            plt.plot(indexed_lats, gauss, color = (0,1,0), label = "gaussian")
            plt.vlines(zeros, ymin = np.min(indexed_if), ymax = np.max(indexed_if), color = (0,0,1), label = "detected zero")
            plt.vlines(ret_zero, ymin = np.min(indexed_if), ymax = np.max(indexed_if), color = (0,1,1), label = "detected zero in range")
            # plt.plot(unique_vals, average_values, color = (0,0,1))
            plt.legend()
            self.show()
        if len(ret_zero) > 1 and np.std(ret_zero) > 5:
            return None
        else:
            return np.mean(ret_zero)
    def analyze_wave(self, projected_image, projected_lat, projected_lon, band):
        # 6 seems to be best shift value
        shift_amount= 6
        
        lon_crop = projected_image[:,self.columns_with_data]
        if self.figures["lon_cropped_proj"]:
            plt.imshow(lon_crop, cmap = "gray")
            self.show()
        
        shifted_image, nan_rows = self.shift_image(projected_image, shift_amount, False)
        shifted_image = self.compare_shifted_and_original(shifted_image, projected_image)
        
        if self.figures["full_shifted_proj"]:
            plt.imshow(shifted_image, cmap = "gray")
            self.show()

        if self.figures["lon_cropped_shifted_proj"]:
            plt.imshow(shifted_image[:,self.columns_with_data], cmap = "gray")
            self.show()

        if self.figures["lon_cropped_shifted_proj_gauss"]:
            gauss = self.compare_shifted_and_original(gaussian_filter(shifted_image,sigma = 3), projected_image)[:,self.columns_with_data]
            self.show()
        
        
        gauss = self.compare_shifted_and_original(gaussian_filter(shifted_image,sigma = 1), projected_image)[:,self.columns_with_data]
        global_det = self.global_determination(gauss, projected_lat[:,self.columns_with_data], projected_lon[:,self.columns_with_data], band)
        local_dets, local_det = self.local_determination(shifted_image[:,self.columns_with_data], projected_lat[:,self.columns_with_data], projected_lon[:,self.columns_with_data], band, global_det)
        # for column in range(projected_image.shape[1]):
        if band >= 118:
            x = 0
        if global_det is None:
            global_det = local_det
        det = global_det + 4*local_det; det /= 5
        return det, [global_det, local_dets, local_det]
        
    def analyze_all_wavelengths(self):
        threshold_of_rows = 0.6
        #keep all the columns that have more than 60% of the rows with data

        projected_image, (projected_lon, projected_lat), _, _= self.equirectangular(self.cube_vis, 1)
        
        #processes run on first projected image
        self.columns_with_data = np.sum(~np.isnan(projected_image), axis = 0) > threshold_of_rows * projected_image.shape[0]
        lats = []
        cum_start_time = time.time()
        index = 0
        leng =  len(self.usable_bands)
        for band in self.cube_vis.bands:
            if int(band) not in self.usable_bands:
                continue
            start_time = time.time()    
            if self.figures["original_image"]:
                fig = plt.figure(self.figure_keys["original_image"])
                plt.imshow(self.cube_vis[int(band)], cmap = "gray")
                plt.title("Cube from "+ self.cube_vis.flyby.name  +" at " +str(band) + "_"+str(self.cube_vis.w[band]) + " µm")
                self.show()
            
            projected_image, (projected_lon, projected_lat), _, _= self.equirectangular(self.cube_vis, band)
            
            lat, shit = self.analyze_wave(projected_image,projected_lat, projected_lon, band)
            lats.append(lat)
            index +=1
            print("processed band", index, "of",leng, "total bands | took", time.time() - start_time, "seconds | expected time left:", (time.time() - cum_start_time) * (leng / index -1) , "so total time is", (time.time() - cum_start_time) * (leng / index), end = "\r")
        
        for band in self.cube_ir.bands:
            if int(band) not in self.usable_bands:
                continue
            start_time = time.time()            
            if self.figures["original_image"]:
                fig = plt.figure(self.figure_keys["original_image"])
                plt.imshow(self.cube_ir[int(band)], cmap = "gray")
                plt.title("Cube from "+ self.cube_ir.flyby.name  +" at " +str(band) + "_"+str(self.cube_ir.w[band]) + " µm")
                self.show()
            
            projected_image, (projected_lon, projected_lat), _, _= self.equirectangular(self.cube_ir, band)
            
            lat, shit = self.analyze_wave(projected_image,projected_lat, projected_lon, band)
            lats.append(lat)
            index +=1
            print("processed band", index, "of",leng, "total bands | took", time.time() - start_time, "seconds | expected time left:", (time.time() - cum_start_time) * (leng / index -1) , end = "\r")

        return {"nsa_latitude":np.mean(lats), "latitude_array": lats}


"""
52,53,63,70,97,98,102,103,104,105,124,125,127,128,129,130,131,132,133,150
71,72,73,74,75,76,77,78,85,86,87,88,89,90,105,112,113,114,115,116,117,118

"""