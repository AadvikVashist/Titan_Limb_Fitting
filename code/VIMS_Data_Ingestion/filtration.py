from datetime import datetime
import numpy as np
import pandas as pd
import pickle
from thefuzz import fuzz
import os
import re
import webbrowser
import time

def convert_dates(date):
    date_list = date.replace(":", "-").split("-")
    date_list = [d for d in date_list if len(d) > 0]
    date = '-'.join(date_list[0:3])
    if len(date_list) == 3:
        date += "-00:00:00"
    else:
        addition = ':' + ':'.join((6 - len(date_list))*["00"])
        date += "-" + ":".join(date_list[3::]) + addition
    return date


def fix_date(date):
    date = date.replace(" at ", "-").replace("/",
                                             "-").replace(":", "-").split("-")
    date = '-'.join(date)
    return date


def get_limb(csv):
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "limb vis" in h.lower()][0]
    return [header] + [c for c in csv if "yes" in c[header_index].lower()]


def get_percentage(csv, min):
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "percentage" in h.lower()][0]
    return [header] + [c for c in csv[1::] if float(c[header_index]) >= min]


def get_vis(csv):
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "sampling mode" in h.lower()][0]
    header_split_index = [index for index, h in enumerate(
        header[header_index].split("|")) if "vis" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index].lower().split("|")[header_split_index]]


def get_ir(csv):
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "sampling mode" in h.lower()][0]
    header_split_index = [index for index, h in enumerate(
        header[header_index].split("|")) if "ir" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index].lower().split("|")[header_split_index]]


def get_vis_and_ir(csv):
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "sampling mode" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index]]


def get_filtered_phase(csv, phase_list: list):
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "phase" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index] and float(c[header_index]) >= min(phase_list) and float(c[header_index]) <= max(phase_list)]


def get_info_cubes(csv):
    header = csv[0]
    header_indexes = set([index for row in csv for index,
                         c in enumerate(row) if c == "_"])
    return [header] + [row for row in csv[1::] if all([col != "_" for col in [row[i] for i in header_indexes]])]


def get_refined_samples(csv, x, y):
    header = csv[0]
    minx = min(x)
    maxx = max(x)
    miny = min(y)
    maxy = max(y)
    header_index = [index for index, h in enumerate(
        header) if "samples" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index] and int(c[header_index].split("x")[0]) >= minx and int(c[header_index].split("x")[0]) <= maxx and int(c[header_index].split("x")[1]) >= miny and int(c[header_index].split("x")[1]) <= maxy]


def get_targeted_flyby(csv):
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "flyby" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "|" in c[header_index] and "T" in c[header_index].split(" | ")[0]]


def get_mission(csv, mission_ints):
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "mission" in h.lower()][0]
    mission_list = [c[header_index] for c in csv[1::]]
    # sort based on index of occurence in mission_list
    missions = sorted(list(set(mission_list)), key=mission_list.index)
    if type(mission_ints) == list and len(mission_ints) > 1:
        for index in range(len(mission_ints)):
            if type(mission_ints[index]) == str:
                mission = [i for i, m in enumerate(
                    missions) if fuzz.ratio(m, mission_ints[index]) > 70]
                if len(mission) != 1:
                    raise ValueError("Mission name not found")
                else:
                    mission_ints[index] = mission[0]
        return [header] + [c for c in csv[1::] if any([missions[miss] in c[header_index] for miss in mission_ints])]
    else:

        if type(mission_ints) == list and len(mission_ints) == 1:
            mission_ints = mission_ints[0]
        if type(mission_ints) == str:
            mission = [i for i, m in enumerate(
                missions) if fuzz.ratio(m, mission_ints) > 70]
        if len(mission) != 1:
            raise ValueError("Mission name not found")
        else:
            mission_ints = mission[0]
        return [header] + [c for c in csv[1::] if missions[mission_ints] in c[header_index]]


def filter_distance(csv, distance):
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "dist" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index] and max(distance) >= float(c[header_index].replace(",", "").split(" ")[0]) >= min(distance)]


def filter_resolution(csv, resolution):
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "resolution" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index] and max(resolution) >= float(c[header_index].split(" ")[0]) >= min(resolution)]


def filter_dates(csv, start_date, end_date):
    start_date = convert_dates(start_date)
    end_date = convert_dates(end_date)
    start = datetime.strptime(start_date, "%Y-%m-%d-%H:%M:%S")
    end = datetime.strptime(end_date, "%Y-%m-%d-%H:%M:%S")
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "time" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index] and start <= datetime.strptime(fix_date(c[header_index]), "%d-%m-%Y-%H-%M-%S") <= end]


def get_filtered_res(csv, resolution):
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "samples lines" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index] and int(c[header_index].split("x")[0]) >= resolution and int(c[header_index].split("x")[1]) > resolution]


def get_limb(csv, limb_or_no):
    if limb_or_no:
        limb_or_no = "Yes"
    else:
        limb_or_no = "No"
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "limb visible" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index] and c[header_index] == limb_or_no]


def filter_km_pixel(csv, km_pixel: int):
    # filter by the resolution (km/pixel).
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "mean resolution" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index] and int(c[header_index].split(" ")[0]) <= km_pixel]


def get_equatorial_latitude(csv, latitudes: list, na_accepted=False):
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "sub-spacecraft point" in h.lower()][0]
    pattern = r'(-?\d+) N'
    if na_accepted:
        return [header] + [c for c in csv[1::] if "n/a" in c[header_index] or re.search(pattern, c[header_index]).group(1) and int(re.search(pattern, c[header_index]).group(1)) > latitudes[0] and int(re.search(pattern, c[header_index]).group(1)) <= latitudes[1]]
    else:
        return [header] + [c for c in csv[1::] if "n/a" not in c[header_index] and re.search(pattern, c[header_index]).group(1) and int(re.search(pattern, c[header_index]).group(1)) > latitudes[0] and int(re.search(pattern, c[header_index]).group(1)) <= latitudes[1]]


def filter_cubes_by_name(csv, names: list):
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "name" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index] and any([name in c[header_index] for name in names])]


def select_cubes_based_on_images(cubes):
    base = "https://vims.univ-nantes.fr/cube/"
    header = cubes[0]
    name_index = [index for index, h in enumerate(
        header) if "name" in h.lower()][0]

    for cube in cubes[1::]:
        cube_name = cube[name_index]
        webbrowser.open_new_tab(base + cube_name)
        #1
        # 4/8 1519673575_1
        #1
        #1
        #1/4
        #1
        #2/2
        #1
        while True:
            inp = input("Do you want to keep this cube? (y/n)")
            if inp.lower() == "" or inp.lower() == "y":
                break
            elif inp.lower() == "n":
                cubes.remove(cube)
                break
    return cubes


def get_names_of_cubes(cubes):
    header = cubes[0]
    name_index = [index for index, h in enumerate(
        header) if "name" in h.lower()][0]
    return [cube[name_index] for cube in cubes[1::]]


def get_square_cubes(csv):
    header = csv[0]
    header_index = [index for index, h in enumerate(
        header) if "samples lines" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index] and c[header_index].split("x")[0] == c[header_index].split("x")[1]]

def get_flyby_from_cube_name(data, cube_name):
    if "C" in cube_name:
        cube_name = cube_name[1::]
    val = None
    for row in data:
        if row[0] == cube_name:
            val = row[11]
    return val


def get_cubes_with_flyby(cubes, flybys):
    header = cubes[0]
    name_index = [index for index, h in enumerate(
        header) if "name" in h.lower()][0]

    for cube in cubes[1::]:
        cube_name = cube[name_index]
        if cube_name in flybys:
            cubes.remove(cube)
    return cubes

def remove_ones_with_existing_flybys(data, flybys):
    header = data[0]
    header_index = [index for index, h in enumerate(header) if "flyby" in h.lower()][0]
    return [row for row in data if row[header_index] not in flybys]


def select_best_per_flyby(data):
    base = "https://vims.univ-nantes.fr/cube/"
    dictionary = {}
    header_index =  [index for index, h in enumerate(refined_search[0]) if "flyby" in h.lower()][0]
    for cube in refined_search[1::]:
        if cube[header_index] not in dictionary:
            dictionary[cube[header_index]] = [cube]
        else:
            dictionary[cube[header_index]].append(cube)
            
    selections = []
    for key, value in dictionary.items():
        if len(value) == 1:
            selections.append(value[0])
            continue
        else:
            for cube in value:
                cube_name = cube[0]
                webbrowser.open_new_tab(base + cube_name)
                time.sleep(0.5)
            while True:
                inp = input("What cube do you want (give index)?")
                try:
                    inp = int(inp)
                    try:
                        selections.append(value[inp])
                        break
                    except:
                        print("Invalid index, max is " + str(len(value) - 1))
                except:
                    print("Invalid input, must be an integer")
        v = value
        # while True: T59 IS REALLY BAD
        #     inp = input("Do you want to keep this cube? (y/n)")
        #     if inp.lower() == "" or inp.lower() == "y":
        #         break
        #     elif inp.lower() == "n":
        #         cubes.remove(cube)
        #         break
        dictionary[key] = sorted(dictionary[key], key=lambda x: x[0])
    dictionary = dictionary
# Use a list comprehension to extract the values from the list of strings
if __name__ == "__main__":
    with open(os.path.join('/'.join(__file__.split("/")[0:-1]), "data/combined_nantes.pickle"), "rb") as f:
        # Use pickle to dump the variable into the file
        data = pickle.load(f)




    # existing_cubes = ["C1477437155_1",
    # "C1487070016_1",
    # "C1509087583_1",
    # "C1519673575_1",
    # "C1559103132_1",
    # "C1560489660_1",
    # "C1560494730_1",
    # "C1561892179_1",
    # "C1629935228_1",
    # "C1629974346_1",
    # "C1634084887_1",
    # "C1649224228_1",
    # "C1654504364_1",
    # "C1681931457_1",
    # "C1702548750_1",
    # "C1702617349_1",
    # "C1721931258_1",
    # "C1826042588_1",
    # ]
    # #C1654504364_1 is bad
    # #1702617349_1 is bad
    # cube_flybys = [get_flyby_from_cube_name(data, cube) for cube in existing_cubes]
    
    
    
    
    
    # limb = get_limb(data)
    refined_search = get_info_cubes(data)
    # refined_search = get_targeted_flyby(refined_search)
    refined_search = get_filtered_phase(refined_search, [0, 20])
    refined_search = get_vis_and_ir(refined_search)
    # refined_search = get_filtered_res(refined_search, 30)
    refined_search = get_limb(refined_search, True)
    refined_search = filter_km_pixel(refined_search, 200)
    search_with_na = get_equatorial_latitude(refined_search, [-15, 15], True)
    
    # refined_search = remove_ones_with_existing_flybys(refined_search, cube_flybys)
    # move the first row back to its original location
    # header = refined_search[0]
    # refined_search = refined_search[1::]
    # refined_search.reverse()
    # refined_search.insert(0, header)

    # move the first element back to its original location
    refined_search = select_best_per_flyby(refined_search)
    searched = get_names_of_cubes(refined_search)
    print(searched)
    

    # square_cubes = get_square_cubes(refined_search)


    # searched = get_names_of_cubes(square_cubes)

    # limb brightening
    # latitudonal strcutre
    # IR wavelengths and spherical mapping
    # Deposits of lightening

    # lightening should look like the same as earth, a plasma nitrogen, should be all wavelengths
"""
1. Get all the cubes without despiker
2. Analysis. 
3.

"""
