

# from fitting_code.ingestion import main as ingestion
# from fitting_code.data_processing import main as data_processing

from fitting_code.plot_generation import main as plot_generation

from fitting_code.global_analysis import main as global_analysis


# from fitting_code.paper_generation import main as paper_generation




    
if __name__ == '__main__':
    # ingestion.ingest_all()
    # data_processing.run_all_limb_processing(multiprocess=True, emission_cutoff=10, nsa=True)
    # data_processing.process_nsa_data()    
    # data_processing.process_srtc_data()
    global_analysis.derive_all_trends(devEnvironment=True)



    plot_generation.gen_all_plots(devEnvironment=True, multi_process=False)
    
    


    
    # detector = transition_wave(False)
    # detector.run_transitional_detector_all()
    

