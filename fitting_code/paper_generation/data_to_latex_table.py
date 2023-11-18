from settings.get_settings import join_strings, check_if_exists_or_write, SETTINGS
import datetime
table_location = join_strings(
    SETTINGS["paths"]["latex"]["latex_path"], SETTINGS["paths"]["latex"]["table_path"])
CUBES = check_if_exists_or_write(
    SETTINGS["paths"]["cube_json"], base=SETTINGS["paths"]["parent_path"], file_type="json")


def get_table(file_path: str = table_location):
    table = check_if_exists_or_write(file_path, save=False, verbose=True)
    return table


def parse_table_for_cols_and_gen(table: list):
    column_start = [index for index, line in enumerate(
        table) if "COLSTART" in line]
    column_end = [index for index, line in enumerate(
        table) if "COLSTOP" in line]
    if len(column_start) == 0 or len(column_end) == 0:
        raise ValueError(
            "Table does not have column start or end (check which lines have COLSTART and COLSTOP)")
    col_lines = table[column_start[0]+1:column_end[0]]
    # remove all random stuff and get the column names, which are denoted by % at the end of the line
    col_parameters = [col.split("%")[-1].strip()
                      for col in col_lines if not col.strip().startswith("%")]

    table_start = [index for index, line in enumerate(
        table) if "GENSTART" in line]
    table_end = [index for index, line in enumerate(
        table) if "GENSTOP" in line]
    if len(table_start) == 0 or len(table_end) == 0:
        raise ValueError(
            "Table does not have table start or end (check which lines have GENSTART and GENSTOP)")
    return col_parameters, table_start, table_end


def retrieve_cubes(location: str = CUBES):
    selected_cubes = CUBES["selected"]
    # for flyby, selected_cube in selected_cubes.items():
    #     selected_cube = "C"+selected_cube
    return selected_cubes


def parse_ingestion():
    data = check_if_exists_or_write(join_strings(SETTINGS["paths"]["parent_path"], SETTINGS["paths"]["ingestion"]
                                    ["ingestion_path"], SETTINGS["paths"]["ingestion"]["nantes_file_location"]), save=False, verbose=True)
    if data is None:
        raise ValueError("Data is None, check if file exists")
    return data


def cross_validate(database: list, cubes: dict):
    database_cubes = [dat[0] for dat in database]
    for flyby, cube in cubes.items():
        cube_index = database_cubes.index(cube)
        cube = database[cube_index]
        cubes[flyby] = cube
    cubes["KEY"] = database[0]
    return cubes


def add_time_to_cube(cubes: dict):
    time_index = [index for index, val in enumerate(
        cubes["KEY"]) if "time" in val.lower()][0]
    for flyby, cube in cubes.items():
        if flyby == "KEY":
            continue
        cube_time = cube[time_index]
        append = get_time(cube_time)
        cube.extend(append[0:-1])
        cubes[flyby] = cube
        key_append_vals = append[-1]
    cubes["KEY"].extend(key_append_vals)
    return cubes


def get_time(date_time: str):

    # Get the start of the year for the same year as the given datetime
    datetime_var = datetime.datetime.strptime(
        date_time, '%d/%m/%Y at %H:%M:%S').replace(tzinfo=datetime.timezone.utc)

    start_of_year = datetime.datetime(
        datetime_var.year, 1, 1, tzinfo=datetime.timezone.utc)

    # Calculate the time difference in seconds between the given datetime and the start of the year
    time_difference_seconds = (datetime_var - start_of_year).total_seconds()

    # Calculate the total number of seconds in a year (considering leap years)
    total_seconds_in_year = 366 * 24 * 60 * \
        60 if datetime_var.year % 4 == 0 else 365 * 24 * 60 * 60

    # Calculate the percentage of the year
    percentage_of_year = (time_difference_seconds / total_seconds_in_year)
    return datetime_var.year, datetime_var.month, datetime_var.day, datetime_var.year + percentage_of_year, ["Year", "Month", "Day", "Year Percentage"]


def generate_mapping(data_list):
    mapping = {}
    for index, value in enumerate(data_list):
        mapping[f"%{index}"] = value
    return mapping


def format_cube_data_for_table(table: list, parameters: list, generation_start: int, generation_end: int,  cubes: dict):
    """
    cube : Name
    target : Target
    date : Image mid-time 
    sample_res: Samples Lines
    sub_instrument : Sampling Mode (VIS | IR)
    exposure: Exposure (VIS | IR)
    obs_seq: Observation Sequence
    seq : Sequence
    rev : Revolution
    orbit : Orbit
    mission: Mission
    flyby: Flyby
    distance: Distance
    spacial_res: Mean resolution
    sub_spacecraft_lat: Sub-Spacecraft point
    sub_spacecraft_lon: Sub-Spacecraft point
    sub_spacecraft: Sub-Spacecraft point
    sub_solar_lat: Sub-Solar point
    sub_solar_lon: Sub-Solar point
    sub_solar: Sub-Solar point
    incidence: Incidence (min | max)
    emergence: Emergence (min | max)
    phase: Phase
    limb: Limb visible
    loc_x: x
    loc_y: y
    loc_radius: radius
    loc_percentage: percentage
    year: Year
    month: Month
    day: Day
    year_perc: Year Percentage
    """

    MAPPING = {
        'cube': 'Name',
        'target': 'Target',
        'date': 'Image mid-time',
        'sample_res': 'Samples Lines',
        'sub_instrument': 'Sampling Mode (VIS | IR)',
        'exposure': 'Exposure (VIS | IR)',
        'obs_seq': 'Observation Sequence',
        'seq': 'Sequence',
        'rev': 'Revolution',
        'orbit': 'Orbit',
        'mission': 'Mission',
        'flyby': 'Flyby',
        'distance': 'Distance',
        'spatial_res': 'Mean resolution',
        'sub_spacecraft_lat': 'Sub-Spacecraft point',
        'sub_spacecraft_lon': 'Sub-Spacecraft point',
        'sub_spacecraft': 'Sub-Spacecraft point',
        'sub_solar_lat': 'Sub-Solar point',
        'sub_solar_lon': 'Sub-Solar point',
        'sub_solar': 'Sub-Solar point',
        'incidence': 'Incidence (min | max)',
        'emergence': 'Emergence (min | max)',
        'phase': 'Phase',
        'limb': 'Limb visible',
        'loc_x': 'x',
        'loc_y': 'y',
        'loc_radius': 'radius',
        'loc_percentage': 'percentage',
        'year': 'Year',
        'month': 'Month',
        'day': 'Day',
        'year_perc': 'Year Percentage'
    }
    cube_key = cubes["KEY"]
    mapping_indices = {key: cube_key.index(value) for key,value in MAPPING.items()}
    # parameters = [MAPPING[parameter.lower()] for parameter in parameters]
    new_table=[]
    for flyby, cube_data in cubes.items():
        if flyby == "KEY":
            continue
        params = []
        for parameter in parameters:
            parameter = parameter.lower().strip()
            param_val = cube_data[mapping_indices[parameter]]
            if parameter == "sub_spacecraft_lat" or parameter == "sub_solar_lat":
                param_val = param_val.split("|")[0].strip()
            elif parameter == "sub_spacecraft_lon" or parameter == "sub_solar_lon":
                param_val = param_val.split("|")[1].strip()
            elif "flyby" in parameter:
                param_val =  param_val.split("|")
                if len(param_val) == 1:
                    param_val = "$" + param_val[0].strip() + "^{1}$"
                else:
                    param_val = param_val[0].strip()
            elif "spatial_res" in parameter:
                param_val = param_val.split(" ")[0].strip()
            params.append(str(param_val))

            # else:
            #     raise ValueError(f"Parameter {parameter} not found")
        new_table.append(' & '.join(params) + "\\\\")
    new_table = table[0:generation_start[0]+1] + new_table + table[generation_end[0]:]
    
    return new_table


def rewrite_table_with_cubes():
    table_string = get_table()
    table_lines = table_string.split("\n")
    parameters, gen_start, gen_end = parse_table_for_cols_and_gen(table_lines)
    cubes = retrieve_cubes()
    all_cube_data = parse_ingestion()
    selected_cube_data = cross_validate(all_cube_data, cubes)
    selected_cube_data = add_time_to_cube(selected_cube_data)
    new_table = format_cube_data_for_table(
        table_lines, parameters=parameters, generation_start=gen_start, generation_end=gen_end, cubes=selected_cube_data)
    table_string = "\n".join(new_table)
    # review once
    print(*new_table, sep="\n")
    while True:
        review = input("Good table (y/n)? ")
        if review.lower() == "y":
            print("Saving table with check if exists or write")
            check_if_exists_or_write(table_location, data = table_string, save=True, verbose=True, force_write=True)
            break
        elif review.lower() == "n":
            print("Not saving table")
            break
        else:
            print("Invalid input, please try again")
