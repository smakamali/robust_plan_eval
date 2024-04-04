# TODO: expose learning rate scheduler parameters

import os
import pickle
from train import train

results_dir = os.path.join('.','results')

if __name__ == '__main__':
    # # On Windows, call freeze_support() to support freezing the script into an executable
    # from multiprocessing import freeze_support
    # freeze_support()

    architectures = [
                    # 'neo_plus',
                    # 'bao_plus',
                    # 'neo',
                    # 'bao',
                    'roq',
                    ]
    experiment_id = 'ceb_750'
    
    training_time_dict  = {}
    for arch in architectures:

        training_time = train(
            experiment_id = experiment_id,
            architecture_p = arch,
            files_id= 'ceb_750',
            benchmark_files_id = 'job_main',
            labeled_data_dir='./labeled_data',
            max_epochs = 1000,
            patience = 50,
            num_experiments = 5,
            num_workers = 5,
            seed = 2,
            reload_data = True,
            val_samples = 0.1,
            test_samples = 100,
            test_slow_samples = None,
            target = 'latency'
            )
        
        training_time_dict[arch] = training_time
    
    with open(os.path.join(results_dir,'training_time_{}.pkl'.format(experiment_id)), 'wb') as file:
        pickle.dump(training_time_dict, file)