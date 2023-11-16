

from fitting_code.ingestion import main as ingestion

from fitting_code.data_processing import main as data_processing

from plot_generation import main as plot_generation
from global_analysis.transition_period import transition_wave







    
if __name__ == '__main__':
    ingestion.ingest_all()
    data_processing.run_all_limb_processing(multiprocess=3, emission_cutoff=25)
    data_processing.process_nsa_data()
    
    plot_generation.gen_all_plots(devEnvironment=False, multi_process=3)
    
    

    
    detector = transition_wave(False)
    detector.run_transitional_detector_all()
    
    print("Generating previews\n\n")    

