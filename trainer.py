import os
import pickle
from train import train

results_dir = os.path.join('.','results')

if __name__ == '__main__':
    # # On Windows, call freeze_support() to support freezing the script into an executable
    # from multiprocessing import freeze_support
    # freeze_support()

    architectures = [
                    'roq', #'neo_plus','bao_plus',
                    # 'neo','bao',
                     ]
    experiment_id = 'job_syn'
    
    training_time_dict  = {}
    for arch in architectures:

        training_time = train(
            experiment_id = experiment_id,
            architecture_p = arch,
            files_id='job_syn_all',
            labeled_data_dir='./labeled_data',
            max_epochs = 1000, patience = 20, 
            num_experiments = 5, num_workers = 5,
            seed = 0, reload_data = False,
            val_samples = 0.1, test_samples = 200
            )
        
        training_time_dict[arch] = training_time
    
    with open(os.path.join(results_dir,'training_time_{}.pkl'.format(experiment_id)), 'wb') as file:
        pickle.dump(training_time_dict, file)