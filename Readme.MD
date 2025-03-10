# Swissper

This repository shares how Whisper can be finetuned on Swiss German data. The code uses accelerate on Huggingface and is also suitable for usage with multiple GPUs. There is also an example script for using on SLURM ```run_script_on_ubelix_accelerate.sh```. 

Current Dataset
1. Archimob Corpus

Steps to set up
1. ```pip install -r requirements.txt```
2. Get Archimob corpus and place under "data/" folder.
3. Run the notebook **Create Dataset** to parse the transcripts and then create a csv file connecting the audio files with the transcripts. 
4. Update the paths to the csv file and the audio folder in the script ```finetune_ubelix_accelerate_handle_oom.py``` and run
5. If the script is to be run on Ubelix, change the required parameters in the script ```slurm.sh``` such as the partictions and GPUS and ```num_processes``` in the accelerate command.


Steps in the SwissperX project

1. Collect the datasets [Swiss Parliament Corpus ](https://cs.technik.fhnw.ch/i4ds-datasets) and [Swiss Dial Datasets](https://mtc.ethz.ch/publications/open-source/swiss-dial.html)
2. Preprocess all three datasets to a format linking (audio file location, id number and transcriptions in swiss standard german and standard german). Pay attention that all the datasets have a swiss standard german.
3. Create pipeline for training swissper X model.
    1. Fork and update SwissperX to the latest version.
    2. Format the datasets to the right format
    3. Get a swissper model : Compare performance of different models (own trained model (whisper small as of now), [whisper-large-v3-peft-swiss-german](https://huggingface.co/Flurin17/whisper-large-v3-peft-swiss-german, [whisper-large-v2-swiss-german](notebotIE/whisper-large-v2-swiss-german))) on test sets from all three datasets. 
    4. Get a phoneme model (ASR Wav2VEc model for swiss german). Compare and evaluate existing offerings such as [scasutt/wav2vec2-large-xlsr-52_Swiss_German](https://huggingface.co/scasutt/wav2vec2-large-xlsr-52_Swiss_German), [manifoldix/xlsr-sg-lm](https://huggingface.co/manifoldix/xlsr-sg-lm), [Sango asr_wav2vec2_large_xlsr_53](https://sparknlp.org/2022/09/26/asr_wav2vec2_large_xlsr_53_swiss_german_gpu_sg.html). 
    5. Finetune swissperX version
    6. Evaluate end -to-end performance
4. Containerise and serve.