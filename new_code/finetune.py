from transformers import WhisperTokenizer, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from data_processor import DataProcessor
from whisper import main

parser = argparse.ArgumentParser(description ='add tasks to Whisper')
parser.add_argument('--task', type = str, default='speechsum')
parser.add_argument("--method",type=str, default='token', help="two finetuning methods: token, prompt")
parser.add_argument("--prompt",type=str, help="two finetuning methods: token, prompt")

args = parser.parse_args()








if __name__ == '__main__':
    print('Task:', args.task)
    print('Method', args.method)
    dp = DataProcessor()
    data = dp.load_and_process_data(args.task, num_rows=40000)
    
    main(args, data)
    
