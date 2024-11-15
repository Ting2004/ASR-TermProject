from transformers import WhisperTokenizer, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate


# parser = argparse.ArgumentParser(description ='add tasks to Whisper')
# parser.add_argument('--task', metavar ='N', type = str, default='speechsum', nargs ='+',required=True)
# parser.add_argument("--method",type=str, required=True, default='token', help="two finetuning methods: token, prompt")
#parser.add_argument("--prompt",type=str, default='Summerize Speech', help="two finetuning methods: token, prompt")

# args = parser.parse_args()


def add_tokens():
    pass
def add_prompt():
    pass

def fine_tune():
    pass

def main(args):
    ckpt = "openai/whisper-small.en"
    tokenizer = WhisperTokenizer.from_pretrained(ckpt,language="English")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(ckpt)
    processor = WhisperProcessor.from_pretrained(ckpt, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(ckpt)
    
    if args.method=='token':
        new_tokens = ['<|speechsum|>', '<|noisedet|>', '<|sentdet|>', '<|agegender|>']
        processor.tokenizer.add_special_tokens(dict(additional_special_tokens=new_tokens))
        processor.tokenizer.set_prefix_tokens(task=args.task)
        processor.tokenizer.add_tokens(list(new_tokens))
        model.resize_token_embeddings(len(processor.tokenizer))
    elif args.method=='prompt':
        processor.tokenizer.task= 'startoflm'
        processor.tokenizer.prompt = processor.tokenizer("new prompt added").input_ids
        
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-hi",  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=5000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model("./whisper-small-hi")
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
