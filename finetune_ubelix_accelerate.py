#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
from datasets import load_dataset, Audio, Dataset
import pandas as pd
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
import evaluate
import torch
from transformers import WhisperForConditionalGeneration
from transformers import WhisperProcessor
from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import logging
import time
from transformers import pipeline

from accelerate import Accelerator



metric = evaluate.load("wer")



formatter = logging.Formatter('%(asctime)s %(lineno)s %(levelname)s %(message)s')
def setup_logger(name, log_file="app_acc.log", level=logging.INFO):

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger



def compute_metrics_test_lower(pred_str_list, label_str_list):
    pred_str_list = [string.lower() for string in pred_str_list]
    label_str_list = [string.lower() for string in label_str_list]
    wer = 100 * metric.compute(predictions=pred_str_list, references=label_str_list)
    return {"wer": wer}


def prepare_dataset(batch, feature_extractor, tokenizer):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    del batch["audio"]
    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    del batch["sentence"]
    return batch


logger = setup_logger("logger")
logger.info(f'Is Cuda available? {torch.cuda.is_available()}')


#device = "cuda" if torch.cuda.is_available() else "cpu"
#print(device)


df = pd.read_csv("audio_df_umlauts.csv")


# In[10]:


logger.info(f'Dataset loaded. Size of dataset {len(df)}')



#n = 0.5
#audio_dataset = Dataset.from_pandas(df.iloc[0:round(n*len(df))]).cast_column("audio", Audio())
audio_dataset = Dataset.from_pandas(df).cast_column("audio", Audio())


audio_dataset = audio_dataset.train_test_split(test_size=0.2, seed=42)
temp_ds = audio_dataset["test"].train_test_split(test_size = 0.5, seed=42)
audio_dataset["valid"]=temp_ds["train"]
audio_dataset["test"]=temp_ds["test"]

logger.info(f'Dataset is split into train, test and validation. Size of train dataset {len(audio_dataset["train"])}')
logger.info(f'Size of train dataset {len(audio_dataset["train"])}')
logger.info(f'Size of validation dataset {len(audio_dataset["valid"])}')
logger.info(f'Size of test dataset {len(audio_dataset["test"])}')

model_name = "openai/whisper-small"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

logger.info(f'Model name: {model_name}')

tokenizer = WhisperTokenizer.from_pretrained(model_name, language="German", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name, language="German", task="transcribe")
audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16000))


audio_dataset_mapped = audio_dataset.map(prepare_dataset, remove_columns=audio_dataset.column_names["train"],# num_proc=4, 
                                  fn_kwargs ={"feature_extractor":feature_extractor, "tokenizer": tokenizer})
logger.info(f'Dataset mapped to create features')
model = WhisperForConditionalGeneration.from_pretrained(model_name)#.to(device)


torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def do_evaluation(data): 
    label_str_list = data["sentence"]
    result = []
    batch_size = 8
    for batch_num in tqdm(range(len(data) // batch_size + 1)):
        audio_files_list = [item.get("path") for item in data["audio"][batch_size * batch_num: (batch_num+1) * batch_size]]
        res = pipe(audio_files_list)
        result += res
    result = [item.get("text") for item in result]
    return compute_metrics_test_lower(pred_str_list=result, label_str_list = label_str_list)




#test_results_before_finetuning = do_evaluation(audio_dataset["test"]) 
#validation_results_before_finetuning = do_evaluation(audio_dataset["valid"]) 
#train_results_before_finetuning = do_evaluation(audio_dataset["train"])


#logger.info(f"Training evaluation results before finetuning {train_results_before_finetuning}")
#logger.info(f"Validation evaluation results before finetuning {validation_results_before_finetuning}")
#logger.info(f"Test evaluation results before finetuning {test_results_before_finetuning}")


model.generation_config.language = "german"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# In[63]:


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# In[ ]:

model.gradient_checkpointing_enable()

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-german_acc_100",  # change to a repo name of your choice
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs = 5,
    logging_steps=100,
    eval_steps=100,
    #max_steps=1000,
    #gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=audio_dataset_mapped["train"],
    eval_dataset=audio_dataset_mapped["valid"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)


processor.save_pretrained(training_args.output_dir)
logger.info(f"Finetuning started")
start_time = time.time()


train_result = trainer.train()
train_results = [ item for item in trainer.state.log_history if item.get("loss")]
validation_results = [ item for item in trainer.state.log_history if item.get("eval_loss")]
logger.info(f'Training logs') 
logger.info(f'\n{pd.DataFrame(train_results).to_markdown()}') 
logger.info(f'Validation logs') 
logger.info(f'\n{pd.DataFrame(validation_results).to_markdown()}') 
logger.info(f"Finetuning ended")
logger.info(f"Time taken to finetune { time.time() -  start_time}")
logger.info(f"Training metrics {train_result}") 
test_results = do_evaluation(audio_dataset["test"]) 
#logger.info(f"Training evaluation results after finetuning {train_results}")
#logger.info(f"Validation evaluation results after finetuning {validation_results}")
logger.info(f"Test evaluation results after finetuning {test_results}")

#diff_train = float(train_results_before_finetuning.get("wer")) - float(train_results.get("wer"))
#diff_valid = float(validation_results_before_finetuning.get("wer")) - float(validation_results.get("wer")) 
#diff_test = float(test_results_before_finetuning.get("wer")) - float(test_results.get("wer"))

#logger.info(f"Finetuning improved results by {diff_train} for training set")
#logger.info(f"Finetuning improved results by {diff_valid}for validation set")
#logger.info(f"Finetuning improved results by {diff_test} for test set")

# In[ ]:




