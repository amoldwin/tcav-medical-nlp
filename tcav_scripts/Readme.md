The script for running TCAV experiments.

To simplify the process of running all the experiments, I made "generate_bash_scripts.ipynb" to automatically generate the arguments for "tcav_experiment.py". Modify as needed.



Depending on several parameters, including where you would like to store your files,
 you can specify the details of your experiment using the following arguments:
 
 
"--bottleneck_name": The name of the layer you would like to interpret. Usually this is the name of the pytorch module in 
the model's "module list", but because we are working with some non-standard models, we assign our own names to some model's layers
in tcav/model.py 

NOTE: For all of our experiments, we use the penultimate layer for TCAV. If you want to run TCAV on other layers, you will need to edit "tcav/model.py" to get activations from the desired layer.


"--ckpt_path": the path of the model that we woulk like to interpet


"--working_dir" The path where the output files will be stored


"--activations_dir" The path where activations for individual documents will be stored. These take up a lot of space and are usually not needed after TCAV has been computed, so we recommend using temporary storage for this


"--source_dir" path where data paths come from

"--results_dir": Path to store TCAV results


"--target": The name of the target. For our experiments, we uised "Positive" as the target, because we only looked at binary classification


"--concepts_csv_path":  the path of a csv file with two columns: 1 for concept names, and the other with the names of the files containing negative examples for each concept


"--experiment_name": string to add to output files to remember which experiment it came from


"--data_encode_function_name": Which text encoder to use. See ../tcav/encode_functions.py for these

"--num_trials" We did 50 trials for our experiments, but more is always better if your model is simple enough to make it feasable with your hardware
