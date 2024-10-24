# Swissper

This repository shares how Whisper can be finetuned on Swiss German data. The code uses accelerate on Huggingface and is also suitable for usage with multiple GPUs. There is also an example script for using on SLURM ```run_script_on_ubelix_accelerate.sh```. 

Dataset
1. Archimob Corpus

Steps to set up
1. ```pip install -r requirements.txt```
2. Get Archimob corpus and place under "data/" folder.
3. Run the notebook **Create Dataset** to parse the transcripts and then create a csv file connecting the audio files with the transcripts. 
4. Update the paths to the csv file and the audio folder in the script ```finetune_ubelix_accelerate.py``` and run
5. If the script is to be run on Ubelix, change the required parameters in the script ```run_script_on_ubelix_accelerate.sh``` such as the partictions and GPUS and ```num_processes``` in the accelerate command.