


from fitting_code.plot_generation.generate_previews import generate_cube_previews
from fitting_code.plot_generation.generate_four_figs import gen_quad_plots
from fitting_code.plot_generation.u import gen_u1_u2_figures
from fitting_code.plot_generation.misc_plotting import gen_plots
from fitting_code.plot_generation.save_bands import gen_image_bands
from typing import Union

def gen_all_plots(devEnvironment: bool = True, multi_process: Union[bool,int] = 3):
    bands = gen_image_bands(devEnvironment=devEnvironment)
    bands.gen_all(multi_process=multi_process)
    
    plots = gen_quad_plots(devEnvironment=devEnvironment)
    plots.quad_all(multi_process=multi_process)
    

    
    
    
    plots = gen_u1_u2_figures(devEnvironment=devEnvironment)
    plots.gen_u_vs_time(multi_process=multi_process)
    plots.u1_u2_all_figures(multi_process=multi_process)

    # misc = gen_plots(devEnvironment=devEnvironment)
    # misc.gen_image_overlay(cube_name="C1477437155_1", band=118)
    
    # preview = generate_cube_previews(devEnvironment=devEnvironment)
    # preview.enumerate_all(multi_process=multi_process)