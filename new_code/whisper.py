from transformers import WhisperTokenizer, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from datasets import concatenate_datasets, DatasetDict


# parser = argparse.ArgumentParser(description ='add tasks to Whisper')
# parser.add_argument('--task', metavar ='N', type = str, default='speechsum', nargs ='+',required=True)
# parser.add_argument("--method",type=str, required=True, default='token', help="two finetuning methods: token, prompt")
# parser.add_argument("--prompt",type=str, default='Summerize Speech', help="two finetuning methods: token, prompt")

# args = parser.parse_args()

ckpt = "openai/whisper-small.en" #"./whisper-small-prompt-speechsum"
print('loded model:', ckpt)
tokenizer = WhisperTokenizer.from_pretrained(ckpt,language="English")
feature_extractor = WhisperFeatureExtractor.from_pretrained(ckpt)
processor = WhisperProcessor.from_pretrained(ckpt, language="English", task="transcribe", device_map='cpu')
model = WhisperForConditionalGeneration.from_pretrained(ckpt, device_map="cuda") #WhisperForConditionalGeneration.from_pretrained(ckpt)

def main_eval(args, data):
    if args.method=='token':
        new_tokens = ['<|speechsum|>', '<|noisedet|>', '<|sentdet|>', '<|agegender|>']
        processor.tokenizer.add_special_tokens(dict(additional_special_tokens=new_tokens), replace_additional_special_tokens=False)
        processor.tokenizer.set_prefix_tokens(task=args.task)
        processor.tokenizer.add_tokens(list(new_tokens))
    elif args.method=='prompt':
        processor.tokenizer.task= 'startoflm'
        processor.tokenizer.prompt = processor.tokenizer("new prompt added").input_ids
        
    data= data.train_test_split(test_size=0.3, seed=42)
    
    if args.task == "speechsum":
        result = data["test"].map(map_to_pred_speechsum)   
    elif args.task=="noisedet":
        result = data["test"].map(map_to_pred_noisedet)
        
    elif args.task=="sentdet":
        result = data["test"].map(map_to_pred_sentdet)
    elif args.task=="agegender":
        result = data["test"].map(map_to_pred_agegender)
    else:
        raise ValueError("Task must be one of : [speechsum, noisedet, sentdet, agegender]")
    
    return result

def main(args, data):
    if args.method=='token':
        new_tokens = ['<|speechsum|>', '<|noisedet|>', '<|sentdet|>', '<|agegender|>']
        processor.tokenizer.add_special_tokens(dict(additional_special_tokens=new_tokens), replace_additional_special_tokens=False)
        processor.tokenizer.set_prefix_tokens(task=args.task)
        processor.tokenizer.add_tokens(list(new_tokens))
        model.resize_token_embeddings(len(processor.tokenizer))
    elif args.method=='prompt':
        processor.tokenizer.task= 'startoflm'
        processor.tokenizer.prompt = processor.tokenizer(args.prompt).input_ids
        
    if args.prompt is not None:
        processor.tokenizer.prompt = processor.tokenizer(args.prompt).input_ids
        
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    print('prepare datasets')
    if args.task == "speechsum":
        data = data.map(prepare_dataset_speechsum)
    elif args.task=="noisedet":
        data = data.map(prepare_dataset_noisedet)
    elif args.task=="sentdet":
        data = data.map(prepare_dataset_sentdet)
    elif args.task=="agegender":
        data = data.map(prepare_dataset_agegender)
    else:
        raise ValueError("Task must be one of : [speechsum, noisedet, sentdet, agegender]")
    
    tttt= 'True' if args.prompt != None else 'False'
    save_path= f"/ocean/projects/cis240125p/hbukhari/whisper-small2-{args.method}-{args.task}-{tttt}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_path,  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=8000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(save_path)
    print('Done')
    
def main_final(args, data):
    if args.method=='token':
        new_tokens = ['<|speechsum|>', '<|noisedet|>', '<|sentdet|>', '<|agegender|>']
        processor.tokenizer.add_special_tokens(dict(additional_special_tokens=new_tokens), replace_additional_special_tokens=False)
        processor.tokenizer.add_tokens(list(new_tokens))
        model.resize_token_embeddings(len(processor.tokenizer))
    elif args.method=='prompt':
        processor.tokenizer.task= 'startoflm'
        processor.tokenizer.prompt = processor.tokenizer(args.prompt).input_ids
        
    
        
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    print('prepare datasets')
    
    processor.tokenizer.set_prefix_tokens(task="speechsum")
    data[0] = data[0].map(prepare_dataset_speechsum)
    
    if args.prompt is not None:
        processor.tokenizer.prompt = processor.tokenizer("true , false").input_ids
    processor.tokenizer.set_prefix_tokens(task="noisedet")
    data[1] = data[1].map(prepare_dataset_noisedet)
    
    if args.prompt is not None:
        processor.tokenizer.prompt = processor.tokenizer("Negative , Neutral , Positive").input_ids
    processor.tokenizer.set_prefix_tokens(task="sentdet")
    data[2] = data[2].map(prepare_dataset_sentdet)
    
    processor.tokenizer.set_prefix_tokens(task="agegender")
    data[3] = data[3].map(prepare_dataset_agegender)
    
    #merge
    for i in range(4):
        cols_to_remove = data[i]['train'].column_names
        cols_to_remove.remove("labels")
        cols_to_remove.remove("input_features")
        data[i] = data[i].remove_columns(cols_to_remove)
    
    data_train = concatenate_datasets([data[0]['train'], data[1]['train'], data[2]['train'], data[3]['train']])
    data_test = concatenate_datasets([data[0]['test'], data[1]['test'], data[2]['test'], data[3]['test']])
    data= DatasetDict({"train": data_train,"test": data_test})
    
    #train shit
    tttt= 'True' if args.prompt != None else 'False'
    save_path= f"/ocean/projects/cis240125p/hbukhari/whisper-small-{args.method}-All-{tttt}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=save_path,  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=8000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(save_path)
    print('Done')
    
    
def compute_metrics(pred):
    metric = evaluate.load("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

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
    
    
    
    
    
def map_to_pred_speechsum(batch):

    audio = batch["audio"]

    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features

    #batch["model_input"]= str(tokenizer.decode(tokenizer(batch['summary']).input_ids, skip_special_tokens=False))
    
    batch["reference"] = processor.tokenizer._normalize(batch['summary'])

    
    with torch.no_grad():

        predicted_ids = model.generate(input_features.to("cuda"))[0]

    transcription = processor.decode(predicted_ids)

    batch["prediction"] = processor.tokenizer._normalize(transcription)

    return batch

def map_to_pred_noisedet(batch):

    audio = batch["audio"]

    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features

    batch["reference"] = processor.tokenizer._normalize(batch['text'])


    with torch.no_grad():

        predicted_ids = model.generate(input_features.to("cuda"))[0]

    transcription = processor.decode(predicted_ids)

    batch["prediction"] = processor.tokenizer._normalize(transcription)

    return batch
def map_to_pred_sentdet(batch):

    audio = batch["audio"]

    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features

    batch["reference"] = processor.tokenizer._normalize(batch['sentiment'])


    with torch.no_grad():

        predicted_ids = model.generate(input_features.to("cuda"))[0]

    transcription = processor.decode(predicted_ids)

    batch["prediction"] = processor.tokenizer._normalize(transcription)

    return batch
def map_to_pred_agegender(batch):

    audio = batch["audio"]

    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features

    batch["reference"] = processor.tokenizer._normalize(f'{batch["age"]}, {batch["gender"]}')


    with torch.no_grad():

        predicted_ids = model.generate(input_features.to("cuda"))[0]

    transcription = processor.decode(predicted_ids)

    batch["prediction"] = processor.tokenizer._normalize(transcription)

    return batch

def prepare_dataset_speechsum(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["summary"]).input_ids
    return batch

def prepare_dataset_noisedet(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["noise"]).input_ids
    return batch

def prepare_dataset_sentdet(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["sentiment"]).input_ids
    return batch

def prepare_dataset_agegender(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = processor.tokenizer(f'{batch["age"]}, {batch["gender"]}').input_ids
    return batch



