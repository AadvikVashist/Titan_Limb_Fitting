from settings.get_settings import join_strings, check_if_exists_or_write, SETTINGS
from urllib.request import urlretrieve
import os
CUBES = check_if_exists_or_write(SETTINGS["paths"]["cube_json"], base = SETTINGS["paths"]["parent_path"], file_type="json")




def download_cubes_from_json():
    selected_cubes = CUBES["selected"]
    base = "https://vims.univ-nantes.fr/cube/"
    ending_vis = "_vis.cub"
    ending_ir = "_ir.cub"
    save_path = join_strings(SETTINGS["paths"]["parent_data_path"], SETTINGS["paths"]["cube_sub_path"])
    for flyby, selected_cube in selected_cubes.items():
        selected_cube = "C"+selected_cube
        vis_url = base+selected_cube+ending_vis
        ir_url = base+selected_cube+ending_ir
        save_folder = join_strings(save_path, selected_cube)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(join_strings(save_folder, selected_cube+ending_vis)):
            vis = urlretrieve(vis_url, join_strings(save_folder, selected_cube+ending_vis))
        if not os.path.exists(join_strings(save_folder, selected_cube+ending_ir)):
            ir = urlretrieve(ir_url, join_strings(save_folder, selected_cube+ending_ir))
    x = 0