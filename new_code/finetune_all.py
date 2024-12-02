from transformers import WhisperTokenizer, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse
import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
from data_processor import DataProcessor
from whisper import main_final

parser = argparse.ArgumentParser(description ='add tasks to Whisper')
parser.add_argument("--method",type=str, default='token', help="two finetuning methods: token, prompt")
parser.add_argument("--prompt",type=str, help="two finetuning methods: token, prompt")

args = parser.parse_args()








if __name__ == '__main__':
    print('Task:', "all")
    print('Method', args.method)
    print('prompt', 'True' if args.prompt != None else 'False')
    dp = DataProcessor()
    data1 = dp.load_and_process_data('speechsum', num_rows=20000)
    data2 = dp.load_and_process_data('noisedet')
    data3 = dp.load_and_process_data('sentdet')
    data4 = dp.load_and_process_data('agegender', num_rows=20000)
    data=[data1, data2, data3, data4]
    main_final(args, data)
    
