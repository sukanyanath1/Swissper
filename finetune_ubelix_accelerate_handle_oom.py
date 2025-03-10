#!/usr/bin/env python
# coding: utf-8

import gc
import torch
import time
import logging
from tqdm import tqdm
from datasets import DatasetDict
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer,
    WhisperFeatureExtractor, Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline
)
from accelerate import Accelerator
from typing import Any, Dict, List, Union
from dataclasses import dataclass
import evaluate

# Initialize WER metric
metric = evaluate.load("wer")

# Setup logger
def setup_logger(name, log_file="app_acc.log", level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(lineno)s %(levelname)s %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

logger = setup_logger("logger")

# Initialize Accelerator
accelerator = Accelerator()

# Verify GPU availability
logger.info(f'Is CUDA available? {torch.cuda.is_available()}')
logger.info(f'Number of GPUs available: {torch.cuda.device_count()}')

# Load Whisper model & processor
model_name = "openai/whisper-small"                                                                                                               
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)                                                                           
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="German", task="transcribe")                                                    
processor = WhisperProcessor.from_pretrained(model_name, language="German", task="transcribe")                                                    
                                                                                                                                                  
logger.info(f'Model: {model_name} loaded')                                                                                                        
                                                                                                                                                  
# Loading processed audio data                                                                                                                    
                                                                                                                                                  
audio_dataset_mapped = DatasetDict.load_from_disk("processed_audio_dataset_openai/whisper-small")                                                 
                                                                                                                                                  
# Function to keep only 80% of each split                                                                                                         
def keep_n_percent(dataset_dict, n):                                                                                                              
    return DatasetDict({                                                                                                                          
        split: dataset.select(range(int(len(dataset) * n)))  # Keep only 80%                                                                      
        for split, dataset in dataset_dict.items()                                                                                                
    })                                                                                                                                            
                                                                                                                                                  
# Keep only n% of each split                                                                                                                      
#audio_dataset_mapped = keep_n_percent(audio_dataset_mapped, n=0.5)        

audio_dataset_mapped.set_format("torch", device="cpu") 
                                                                                                                                                  
logger.info(f'Dataset preprocessing complete.')            
logger.info(f'Dataset Loaded. Training size: {len(audio_dataset_mapped["train"])}')                                                                                       

torch.cuda.empty_cache()
gc.collect()

# Load model from checkpoint                                                                                                                      
model = WhisperForConditionalGeneration.from_pretrained(model_name)                                                                               
                                                                                                                                                  
# Reduce memory usage with gradient checkpointing      
model.config.use_cache = False                                                                                                                    
model.gradient_checkpointing_enable()                                                                                                             
                                                                                                                                                  
@dataclass                                                                                                                                        
class DataCollatorSpeechSeq2SeqWithPadding:                                                                                                       
    processor: Any                                                                                                                                
    decoder_start_token_id: int                                                                                                                   
                                                                                                                                                  
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:                                     
        input_features = [{"input_features": feature["input_features"]} for feature in features]                                                  
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")                                                         
        label_features = [{"input_ids": feature["labels"]} for feature in features]                                                               
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")                                                          
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)                                                   
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():                                                                      
            labels = labels[:, 1:]                                                                                                                
        batch["labels"] = labels                                                                                                                  
        return batch                                                                                                                              
                                                                                                                                                  
data_collator = DataCollatorSpeechSeq2SeqWithPadding(                                                                                             
    processor=processor,                                                                                                                          
    decoder_start_token_id=model.config.decoder_start_token_id,                                                                                   
)                                                                                                                                                 
                                                                                                                                                  
def compute_metrics(pred):                                                                                                                        
    pred_ids = pred.predictions                                                                                                                   
    label_ids = pred.label_ids 
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {"wer": 100 * metric.compute(predictions=pred_str, references=label_str)}

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-german_acc_100_checkpoint2",
    per_device_train_batch_size=1,  # Lower batch size to fit memory
    gradient_accumulation_steps=8,  # Increase accumulation to reduce memory load
    evaluation_strategy="no",
    learning_rate=5e-6,#1e-5,
    warmup_steps=500,
    num_train_epochs=5,
    logging_steps=500,
    #eval_steps=500,
    fp16=True,  # Enable mixed precision
    #evaluation_strategy="epoch",
    #per_device_eval_batch_size=1,
    save_steps=1000,
    report_to="none",
    generation_max_length=128,
    push_to_hub=False,
    load_best_model_at_end=False,
    metric_for_best_model="wer",
    remove_unused_columns=False,
    )                                                                                                                                                 
                                                                                                                                                  
# Initialize Trainer                                                                                                                              
trainer = Seq2SeqTrainer(                                                                                                                         
    args=training_args,                                                                                                                           
    model=model,                                                                                                                                  
    train_dataset=audio_dataset_mapped["train"],                                                                                                  
    #eval_dataset=audio_dataset_mapped["valid"],                                                                                                   
    data_collator=data_collator,                                                                                                                  
    compute_metrics=compute_metrics,                                                                                                              
    tokenizer=processor.feature_extractor,                                                                                                        
)                                                                                                                                                 
                                                                                                                                                  
# Move trainer to the accelerator                                                                                                                 
trainer = accelerator.prepare(trainer)                                                                                                            
                                                                                                                                                  
torch.cuda.empty_cache()  # Clears unused memory                                                                                                  
torch.cuda.reset_peak_memory_stats()                                                                                                              
                                                                                                                                                  
# Start Training                                                                                                                                  
logger.info("Finetuning started")                                                                                                                 
start_time = time.time()                                                                                                                          
train_result = trainer.train()                                                                                                                    
                                                                                                                                                  
# Save metrics                                                                                                                                    
logger.info(f"Finetuning complete in {time.time() - start_time} seconds") 

# Test Evaluation                                                                                                                                 
logger.info("Starting test evaluation")                                                                                                           
pipe = pipeline(                                                                                                                                  
    "automatic-speech-recognition",                                                                                                               
    model=model,                                                                                                                                  
    tokenizer=processor.tokenizer,                                                                                                                
    feature_extractor=processor.feature_extractor,                                                                                                
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,                                                                    
)                                                                                                                                                 
                                                                                                                                                  
test_results = compute_metrics(trainer.evaluate(audio_dataset_mapped["test"]))                                                                    
logger.info(f"Test evaluation results: {test_results}") 