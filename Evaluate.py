
from tqdm import tqdm
from datasets import load_dataset, Audio, Dataset, DatasetDict
import pandas as pd
from transformers import WhisperProcessor
from transformers import WhisperTokenizer
from transformers import WhisperFeatureExtractor
from transformers import pipeline
import gc
import torch
import evaluate

metric = evaluate.load("wer")

# Function to keep only 80% of each split
def keep_n_percent(dataset_dict, n):
    return DatasetDict({
        split: dataset.select(range(int(len(dataset) * n)))  # Keep only 80%
        for split, dataset in dataset_dict.items()
    })

model_name = "openai/whisper-large-v3-turbo"#"openai/whisper-small"
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="German", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name, language="German", task="transcribe")
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

audio_dataset_mapped = DatasetDict.load_from_disk("processed_audio_dataset_openai/whisper-small")    
audio_dataset_mapped.set_format("torch", device="cpu")        
audio_dataset_mapped = keep_n_percent(audio_dataset_mapped, n=0.8)


metric = evaluate.load("wer")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

pipe = pipeline(
    "automatic-speech-recognition",
    model="YOUR CHECKPOINT",
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

def compute_metrics_test_lower(pred_str_list, label_str_list):                                                                                    
    pred_str_list = [string.lower() for string in pred_str_list]                                                                                  
    label_str_list = [string.lower() for string in label_str_list]                                                                                
    wer = 100 * metric.compute(predictions=pred_str_list, references=label_str_list)                                                              
    return {"wer": wer}    

def do_evaluation(data):
    label_str_list = data["sentence"]
    result = []
    batch_size = 8
    for batch_num in tqdm(range(len(data) // batch_size + 1)):
        audio_files_list = [item.get("path") for item in data["audio"][batch_size * batch_num: (batch_num+1) * batch_size]]
        res = pipe(audio_files_list)
        result += res
    result = [item.get("text") for item in result]
    df = pd.DataFrame({"test_sentence":label_str_list,
        "transcribed_sentence":result})
    test_results =  compute_metrics_test_lower(pred_str_list=result, label_str_list = label_str_list)
    print(f"Test evaluation results after finetuning {test_results}")
    return df

test_results_df = do_evaluation(audio_dataset_mapped["test"])

