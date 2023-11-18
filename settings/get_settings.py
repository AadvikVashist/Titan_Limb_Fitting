
import os
import os.path as path
import pickle
import cv2
import json
from PIL import Image
import csv
import pandas as pd
def join_strings(*args):
    return os.path.join(*args).replace("\\", "/")


def check_if_exists_or_write(file: str, base: str = None, prefix: str = None, file_type=None, save=False, data=None, force_write=False, verbose=True):
    if base and prefix:
        full_path = join_strings(base, prefix, file)
    elif base:
        full_path = join_strings(base, file)
    elif prefix:
        full_path = join_strings(prefix, file)
    else:
        full_path = file
        
    if file_type is not None and not full_path.endswith(file_type):
        file_type = file_type.strip().replace(".", "")
        full_path += "." + file_type
    if save == True:
        if data is None:
            raise ValueError("Data must be provided to save")
        if verbose:
            print("Saving to: ", full_path)
        directory = os.path.dirname(full_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if force_write == False and os.path.exists(full_path):
            if verbose:
                print("File already exists")
            return False
        if full_path.endswith(".pkl") or full_path.endswith(".pickle"):
            with open(full_path, "wb") as file_object:
                pickle.dump(data, file_object)
            if verbose:
                print("Saved", full_path)
        elif full_path.endswith(".json"):
            with open(full_path, "w") as file_object:
                json.dump(data, file_object)
            if verbose:
                print("Saved", full_path)
        elif full_path.endswith(".csv"):
            with open(full_path, 'w', newline='') as file_object:
                writer = csv.writer(file_object)
                # Write the data to the CSV file
                writer.writerows(data)
            if verbose:
                print("Saved", full_path)
        elif full_path.endswith(".png") or full_path.endswith(".jpg") or full_path.endswith(".jpeg") or full_path.endswith(".tiff") or full_path.endswith(".tif"):
            cv2.imwrite(full_path, data)
            if verbose:
                print("Saved", full_path)
        elif full_path.endswith(".tex") or full_path.endswith(".txt"):
            with open(full_path, 'w') as file_object:
                file_object.write(data)
            if verbose:
                print("Saved", full_path)
        else:
            if verbose:
                print("File type not recognized")
    else:
        if os.path.exists(full_path):
            if full_path.endswith(".pkl") or full_path.endswith(".pickle"):
                with open(full_path, 'rb') as pickle_file:
                    data = pickle.load(pickle_file)
            elif full_path.endswith(".json"):
                with open(full_path, "r") as file_object:
                    data = json.load(file_object)
            elif full_path.endswith(".csv"):
                data = pd.read_csv(full_path)
            elif full_path.endswith(".png") or full_path.endswith(".jpg") or full_path.endswith(".jpeg") or full_path.endswith(".tiff") or full_path.endswith(".tif"):
                data = Image.open(full_path)
            elif full_path.endswith(".tex") or full_path.endswith(".txt"):
                with open(full_path, 'r') as file_object:
                    data = file_object.read()
            else:
                data = None
            return data
        else:
            return False
def get_cumulative_filename(path):
    return SETTINGS["paths"]["cumulative_path"] + SETTINGS["paths"][path] + ".pkl"

SETTINGS = check_if_exists_or_write("settings.json", base =os.path.dirname(os.path.abspath(__file__)), file_type="json", save=False)
SETTINGS = SETTINGS