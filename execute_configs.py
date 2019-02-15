
from helper_classes import DataAnalyser


p='Experiments'
dl_learner_path = '/home/demir/Desktop/physical_embedding/dllearner-1.3.0/bin/cli'

# run DL learner
analyser = DataAnalyser(execute_DL_Learner=dl_learner_path)
analyser.generated_responds(folder_path=p)

# collect reponses
#analyser.collect_data(p)

#analyser.combine_all_data(p)