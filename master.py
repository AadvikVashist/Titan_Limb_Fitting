

from fitting_code.ingestion import main as ingestion

from fitting_code.data_processing import main as data_processing

from fitting_code.plot_generation import main as plot_generation
from fitting_code.global_analysis import main as global_analysis


from fitting_code.paper_generation import main as paper_generation




    
if __name__ == '__main__':
    # ingestion.ingest_all()
    # data_processing.run_all_limb_processing(multiprocess=3, emission_cutoff=25)
    # data_processing.process_nsa_data()
    
    paper_generation.all_paper_gens()
    
    
    global_analysis.derive_all_trends()
    plot_generation.gen_all_plots(devEnvironment=False, multi_process=3)
    
    
    # global_analysis.gen_all_plots(devEnvironment=False, multi_process=3)


    
    # detector = transition_wave(False)
    # detector.run_transitional_detector_all()
    
    print("Generating previews\n\n")    

