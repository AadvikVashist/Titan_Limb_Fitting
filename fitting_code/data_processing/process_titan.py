
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor  # or RandomForestRegressor for regression problems
from sklearn.svm import SVR  # or SVR for regression

from PIL import Image
import os
from settings.get_settings import join_strings, check_if_exists_or_write, SETTINGS, get_cumulative_filename
import cv2
import matplotlib.pyplot as plt 
import pandas as pd
#gaussian filter
from scipy.ndimage import gaussian_filter
from collections import defaultdict
#pchip interpolation
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import PartialDependenceDisplay

#get fit_data
from .fitting import fit_data
data_folder = join_strings(SETTINGS['paths']["parent_data_path"], "SRTC++/v1/")
output_folder = join_strings(SETTINGS['paths']["parent_data_path"], "SRTC++/output/")
listdir = os.listdir(data_folder)
fit_obj = fit_data()


import csv

def write_rows_to_csv(headers, data, file_name):
    """
    Write row data to a CSV file.

    Parameters:
    - headers: A list of strings representing the header names for the columns, can be None if no headers are needed.
    - data: A 2D list where each inner list represents a row with column data.
    - file_name: The name of the CSV file to write to.
    """
    # Open the CSV file for writing
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header row if provided
        if headers is not None:
            writer.writerow(headers)

        # Write the data rows
        writer.writerows(data)

def csv_to_data(file_name):
    """
    Reads a CSV file, converts the first 7 columns to a NumPy array, and stores the last column as a list.

    Args:
        file_name: The name of the CSV file.

    Returns:
        A tuple containing:
            - A NumPy array of the first 7 columns (excluding header).
            - A list of the last column (excluding header).
    """
    # Read the CSV file using pandas
    df = pd.read_csv(file_name)

    # Separate the first 7 columns (excluding header) and convert to NumPy array
    data = np.array(df.iloc[::, :7])

    # Separate the last column (excluding header) and convert to a list
    bools = df.iloc[:, -1].tolist()

    return data, bools
def show_images():
    combined_mask = 0
    total = 0
    for image in listdir:
        # plt.figure()
        if image.endswith('.tif'):
            print(image)
            # img = np.array(img)
            #using opencv, apply a mask to the image
            mask = cv2.imread(join_strings(data_folder, image),0)
            if combined_mask is 0:
                combined_mask = np.zeros(mask.shape)
            mask = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)[1]
            combined_mask += mask
            total += 1
            #show the image
    combined_mask = combined_mask/total*100/255
    final_mask = cv2.threshold(combined_mask, 75, 100, cv2.THRESH_BINARY)[1]
    # plt.imshow(final_mask, cmap='gray')
    # plt.show()    
    return final_mask

final_mask = show_images()






def split_data(input):
    # Assuming 'data' is your 1000x7 numpy array
    #get rid all of input when the 7th column is 0
    X = input[:, :6]  # Select all rows and the first 6 columns as features
    y = input[:, 6]   # Select all rows and the 7th column as the target
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def RandomForest(X_train, X_test, y_train, y_test):
    # Initialize the Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    rf.fit(X_train, y_train)

    # Make predictions
    rf_predictions = rf.predict(X_test)

    # Evaluate the model
    rf_score = rf.score(X_test, y_test)
    print(f"Random Forest accuracy: {rf_score}")
    return rf, rf_score, rf_predictions

def  GradientBoost(X_train, X_test, y_train, y_test):
    # Initialize the GBM model
    gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Train the model
    gbm.fit(X_train, y_train)

    # Make predictions
    gbm_predictions = gbm.predict(X_test)

    # Evaluate the model
    gbm_score = gbm.score(X_test, y_test)
    print(f"GBM accuracy: {gbm_score}")
    return gbm, gbm_score, gbm_predictions

def SVC_algorithm(X_train, X_test, y_train, y_test):
        
    # Initialize the SVM model
    svm = SVR(kernel='rbf')

    # Train the model
    svm.fit(X_train, y_train)

    # Make predictions
    svm_predictions = svm.predict(X_test)

    # Evaluate the model
    svm_score = svm.score(X_test, y_test)
    print(f"SVM accuracy: {svm_score}")
    return svm, svm_score, svm_predictions

#find the center of the mask
def find_center(mask):
    #find the center of the mask
    center = np.array(mask.shape)/2 -0.5
    ##dont include 0's in the mask, get all the non-zero indices
    mask_indices = np.nonzero(mask)
    radius = int(np.sqrt(len(mask_indices[0])/np.pi))
    other_center = np.nanmean(mask_indices, axis=1)
    # plt.imshow(mask, cmap='gray')
    # plt.scatter(center[1], center[0], c='r')
    # plt.scatter(other_center[1], other_center[0], c='b')
    # plt.show()
    return center, other_center,radius, mask.shape[0]


def get_emission(size, radius, center):
    # Image dimensions and sphere radius in pixels

    # Center coordinates
    cx =center[0]
    cy= center[1]

    # Create a grid of x, y coordinates
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)

    # Calculate the distance of each pixel from the center
    d = np.sqrt((X - cx)**2 + (Y - cy)**2)

    # Initialize the emission angle array with 90degs for pixels outside the sphere
    emission_angle = np.full((size, size), np.pi/2)

    # Calculate emission angle for pixels inside the sphere
    inside_sphere = d <= radius
    emission_angle[inside_sphere] = np.arccos(np.sqrt(radius**2 - d[inside_sphere]**2) / radius)

    # Convert emission angle from radians to degrees for better interpretability
    emission_angle_degrees = np.degrees(emission_angle)

    # Plot the emission angle
    
    # plt.figure(figsize=(8, 8))
    # plt.imshow(emission_angle_degrees, cmap='jet')
    # plt.colorbar(label='Emission Angle (degrees)')
    # plt.title('Emission Angle across the Sphere')
    # plt.xlabel('Pixel X')
    # plt.ylabel('Pixel Y')
    # plt.show()
    return emission_angle_degrees

def pchip_interpolation(x, y):
    # Ensure x and y are numpy arrays for the interpolation
    x = np.array(x)
    y = np.array(y)
    
    # Create the PCHIP interpolator
    pchip_interpolator = PchipInterpolator(x, y)
    
    # Generate 1000 equally spaced points between min and max of x
    x_interp = np.linspace(x.min(), x.max(), 1000)
    
    # Evaluate the interpolator at the new points
    y_interp = pchip_interpolator(x_interp)
    
    return x_interp, y_interp


center, other_center, radius, size = find_center(final_mask)
emissions = get_emission(size, radius, other_center)

final_mask_bool = final_mask > 0

def determine_noise_and_brigthness(image):

    eme = emissions[final_mask_bool].astype(np.uint16)
    unique_x = np.unique(eme)
    unique_counts = np.bincount(eme)
    filtered_values = image[final_mask_bool]
    
    average_brightness = np.mean(filtered_values[filtered_values != 0])
    
    result_dict = {key: [filtered_values[i] for i, value in enumerate(eme) if value == key] for key in unique_x}
    updated_dict = {key: np.std(value) * len(value) for key, value in result_dict.items()}
    brightness_arr= [np.mean(value) for key, value in result_dict.items()]
    # plt.title(f"Emission vs Brightness {summed} {average_brightness}")
    # plt.plot(eme, filtered_values, 'o')
    # plt.show()
    summed = 0
    for key, value in updated_dict.items():
        summed += value
    summed = summed/len(filtered_values)

    return summed, average_brightness, brightness_arr

def average_y_for_duplicate_x(x, y):
    # Sort x and y values together
    sorted_pairs = sorted(zip(x, y))
    x_sorted, y_sorted = zip(*sorted_pairs)

    # Dictionary to hold sum and count of y values for each x
    temp_dict = defaultdict(lambda: [0, 0])  # Default [sum of y's, count]

    for xi, yi in zip(x_sorted, y_sorted):
        temp_dict[xi][0] += yi  # Add y value
        temp_dict[xi][1] += 1   # Increment count

    # Prepare final lists
    x_unique = []
    y_avg = []

    for xi, (y_sum, count) in temp_dict.items():
        x_unique.append(xi)
        y_avg.append(y_sum / count)  # Calculate average

    return x_unique, y_avg


def bell_curve(inputs):
    #\left(2x-1\right)^{2}+1
    mean = (np.max(inputs)+np.min(inputs))/2
    rang = np.max(inputs)-np.min(inputs)
    # outputs = (5/rang*(inputs-mean))**2+1
    # output =1/outputs
    outputs = np.ones(len(inputs))
    # plt.plot(inputs, output)
    # plt.show()
    return 
def analyze_srtc_data():
    data = []
    sigma = 20
    # x = Jcube_to_csv(size, size, csv=join_strings(data_folder, "Aadvik0.93_0.01_1.00_0.01_0.01_0.50_0.10_p000_geo.colorCCD.Jcube"))
    for ind, image_string in enumerate(listdir):
        # plt.figure()
        if image_string.endswith('.tif'):
            # print(image_string)
            # img = np.array(img)
            #using opencv, apply a mask to the image
            image = cv2.imread(join_strings(data_folder, image_string),0)
            masked_image = image*final_mask_bool
            std, mean,brightness_arr = determine_noise_and_brigthness(image)
            
            x= emissions.flatten()
            y = masked_image.flatten()
            #remove all the values in x,y when x > 75
            y = y[x<75]
            x = x[x<75]
            #sort by emission
            x,y = zip(*sorted(zip(x,y)))
            ##if there are any repeats in x, average the y values
            
            y2 = gaussian_filter(y, sigma=1)
            x3, y3= average_y_for_duplicate_x(x, y)
            y3 = gaussian_filter(y3, sigma=1)
            
            x_interp, y_interp = pchip_interpolation(x3, y3)
            slope = 1
            
            emission_angles_to_normalized = fit_obj.emission_to_normalized(x_interp)
            fit_data = fit_obj.run_quadratic_limb_darkening(emission_angles_to_normalized, gaussian_filter(y_interp,sigma=sigma), weights = bell_curve(emission_angles_to_normalized))
            fitted_data = fit_obj.quadratic_limb_darkening(emission_angles_to_normalized, fit_data[0]["I_0"], fit_data[0]["u1"], fit_data[0]["u2"])
            

            gaussian_filtered = gaussian_filter(y_interp, sigma=sigma)

            # Calculate R² score
            r2 = r2_score(y_interp, gaussian_filtered)
            # print(f"R² Score: {r2}")

            # Calculate Mean Absolute Error (MAE)
            mae = mean_absolute_error(y_interp, gaussian_filtered)/np.mean(y_interp)
            # print(f"Mean Absolute Error (MAE): {mae}")

            # Calculate Mean Squared Error (MSE)
            mse = mean_squared_error(y_interp, gaussian_filtered)/np.mean(y_interp)
            # print(f"Mean Squared Error (MSE): {mse}")

            # Calculate Root Mean Squared Error (RMSE)
            rmse = np.sqrt(mse)
            # print(f"Root Mean Squared Error (RMSE): {rmse}")
            #make 2 subplots
            
            new_image_string = str(ind) + "_" + image_string[11:image_string.index("_p000")]
            splitted_string = new_image_string.split("_")[0:6]; splitted_string[:] = [float(x) for x in splitted_string]
            print(ind, new_image_string, fit_data[0]["u1"], fit_data[0]["u2"], r2, mae, mse, rmse)
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            plt.title("Image and Fitted Data")
            eme_bool = emissions == 90
            axes[0].imshow(np.where(eme_bool, emissions, masked_image), cmap='gray')
            axes[1].set_ylim(0, 255)
            axes[1].plot(fit_obj.emission_to_normalized(x),y, 'o')
            # plt.plot(x,y2, 'o')
            axes[1].plot(emission_angles_to_normalized, y_interp, label = "interpolated")
            axes[1].plot(emission_angles_to_normalized, gaussian_filtered, label = "data used for fit")
            axes[1].plot(emission_angles_to_normalized, fitted_data, label=" u1: " + str(np.around(fit_data[0]["u1"],2)) + " u2: " + str(np.around(fit_data[0]["u2"],2)))
            axes[1].legend()
            axes[1].set_title(f'R2 = {np.round(r2,3)} MAE = {np.round(mae,3)} MSE = {np.round(mse,3)} RMSE = {np.round(rmse,3)}')
            
            if std > 35 or np.max(brightness_arr) < 60:
                data.append([*splitted_string,fit_data[0]["u1"]+fit_data[0]["u2"], False])
                new_image_string += "_bad"
            else:
                data.append([*splitted_string,fit_data[0]["u1"]+fit_data[0]["u2"], True])
                new_image_string += "_good"

            # plt.show()
            print(new_image_string)
            plt.savefig(join_strings(output_folder, "SRTC++_" + new_image_string + ".png"), dpi=300)
            plt.clf()
            plt.close()
            # surface 

    write_rows_to_csv(["input1","input2", "input3", "input4", "input5", "input6", "output", "usable"], data, 'output.csv')

    return data



def plot_feature_influence(model, X_test, feature_names):
    """
    Plots partial dependence plots for the given features in a model.

    Parameters:
    - model: The trained model.
    - X: The input features data (numpy array or pandas DataFrame).
    - features: A list of integers or strings representing the features' indices or names in 'X'.
    - feature_names: A list of names for the features for labeling purposes.
    """
    
    features = [0,1,2,3,4,5]
    reordered_names = [feature_names[i] for i in features]
    
    pdp_display = PartialDependenceDisplay.from_estimator(model, X_test, features = features, feature_names=reordered_names)
    pdp_display.plot()
    plt.show()

def run():
    data,bools = csv_to_data('output.csv')
    data = data[bools]
    x_train, x_test, y_train, y_test = split_data(data)
    rf_model, rf_score, rf_predictions= RandomForest(x_train, x_test, y_train, y_test)
    gbm_model, gbm_score, gbm_predictions= GradientBoost(x_train, x_test, y_train, y_test)
    svm_model, svm_score, svm_predictions = SVC_algorithm(x_train, x_test, y_train, y_test)
    
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_rmse = np.sqrt(rf_mse)
    rf_r2 = r2_score(y_test, rf_predictions)
    
    gbm_mse = mean_squared_error(y_test, gbm_predictions)
    gbm_rmse = np.sqrt(gbm_mse)
    gmb_r2 = r2_score(y_test, gbm_predictions)
    
    svm_mse = mean_squared_error(y_test, svm_predictions)
    svm_rmse = np.sqrt(svm_mse)
    svm_r2 = r2_score(y_test, svm_predictions)
    
    feature_names = ["Lower Haze", "Lower SSA", "Lower Gas", "Upper Haze", "Upper SSA", "Upper Gas"]
    plot_feature_influence(rf_model,x_test, feature_names)
    plot_feature_influence(gbm_model,x_test, feature_names)
    # plot_feature_influence(svm_model,x_test, feature_names)
    x = 0
    

