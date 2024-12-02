from data_processor import DataProcessor
#from eval import EvalMetric
from whisper import main_eval
import argparse
import pandas as pd
from tqdm import tqdm
parser = argparse.ArgumentParser(description ='add tasks to Whisper')
parser.add_argument('--task', type = str, default='speechsum',required=True)
parser.add_argument("--method",type=str, required=True, default='token', help="two finetuning methods: token, prompt")
parser.add_argument("--prompt",type=str, default='Summerize Speech', help="two finetuning methods: token, prompt")


args = parser.parse_args()





if __name__ == '__main__':
    print('Task:', args.task)
    print('Method', args.method)
    dp = DataProcessor()
    data = dp.load_and_process_data(args.task, num_rows=1000)
    
    results= main_eval(args, data)
    
    #metric=EvalMetric(f"<|{args.task}|>")
    
    ree= {'reference':results['reference'], 'prediction': results['prediction'], 'metric':[]}
    # for r, h in tqdm(zip(results['reference'], results['prediction'])):
    #     ref, hyp, res = metric.evaluate(r, h)
    #     ree['reference'].append(ref)
    #     ree['prediction'].append(hyp)
    #     ree['metric'].append(res)
        # print(f"{ref}|{hyp}|{res}")
    pd.DataFrame(ree).to_csv(f"./eval-{args.method}-{args.task}.csv", index=False)
    
    
    
