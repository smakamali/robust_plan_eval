import os
import pickle
from train import train

results_dir = os.path.join('.','results')

if __name__ == '__main__':
    # # On Windows, call freeze_support() to support freezing the script into an executable
    # from multiprocessing import freeze_support
    # freeze_support()

    architectures = [
        'roq',
        'lero',
        'balsa',
        'bao',
        'neo',
        ]
    experiment_id = 'ceb_1000_x5_s317'
    # experiment_id = 'job_main_1x_wlsh_base'#_s313
    
    training_time_dict  = {}
    for arch in architectures:

        training_time = train(
            experiment_id = experiment_id,
            architecture_p = arch,
            files_id= 'ceb_1000',
            proc_files_id='ceb_1000',
            benchmark_files_id = 'job_v2.2',
            labeled_data_dir='./labeled_data/ceb/',
            max_epochs = 100,
            patience = 10,
            num_experiments = 1,
            num_workers = 5,
            seed = 310,
            reload_data = False,
            num_samples = None,
            val_samples = 0.1,
            test_samples = 0.1,
            test_slow_samples = 0.5,
            target = 'latency'
            )
        
        training_time_dict[arch] = training_time
    
    with open(os.path.join(results_dir,'training_time_{}.pkl'.format(experiment_id)), 'wb') as file:
        pickle.dump(training_time_dict, file)