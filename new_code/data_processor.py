from datasets import load_dataset, Audio, concatenate_datasets, DatasetDict
# import pyarray.dataset as ds
import random


token = "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"

class DataProcessor:
    def prepare_mozilla_dataset(self, batch):
        """Preprocess the Mozilla Common Voice dataset to clean text."""
        transcription = batch["sentence"]

        if transcription.startswith('"') and transcription.endswith('"'):
            transcription = transcription[1:-1]

        if transcription[-1] not in [".", "?", "!"]:
            transcription += "."

        batch["sentence"] = transcription
        return batch

    @staticmethod
    def downsample_dataset(dataset, target_size):
        """Downsamples the dataset to the target size."""
        dataset_size = len(dataset)
        indices = random.sample(range(dataset_size), min(target_size, dataset_size))
        return dataset.select(indices)

    def load_and_process_data(self, task, num_rows=0):
        # Load datasets
        if task =='speechsum':
            sp= ["core", 'test']
            data= load_dataset("komats/mega-ssum", split=sp)
            data= DatasetDict({"train": data[0].select(range(num_rows)) if num_rows else data[0],"test": data[1]})
            data['train']= data['train'].remove_columns(['id', 'summary1', 'summary2', 'summary3'])
            data['test']= data['test'].remove_columns(['id', 'summary1', 'summary2', 'summary3'])
            #data = [(x['audio'], x['summary']) for x in data]
        elif task =='noisedet':
            noise_clean_ds = load_dataset("distil-whisper/librispeech_asr-noise", "test-pub-noise", split=["40", "35", "30", "25"])
            noise_clean_ds = concatenate_datasets(noise_clean_ds)
            noise_clean_ds = noise_clean_ds.add_column("noise", ["false"] * len(noise_clean_ds))
            
            noise_noisy_ds = load_dataset("distil-whisper/librispeech_asr-noise", "test-pub-noise", split=["0", "5", "10", "15"])
            noise_noisy_ds = concatenate_datasets(noise_noisy_ds)
            noise_noisy_ds = noise_noisy_ds.add_column("noise", ["true"] * len(noise_noisy_ds))
            data = concatenate_datasets([noise_clean_ds, noise_noisy_ds])
            data= data.remove_columns(['text', 'id'])
            data= data.train_test_split(test_size=0.2, seed=42)
            if num_rows:
                data['train']= data['train'].select(range(num_rows))
            #data = [(x['audio'], x['text']) for x in data]
        elif task=='sentdet':
            sp= ["train", 'test']
            data= load_dataset("asapp/slue", "voxceleb", split=sp, token="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", trust_remote_code=True)
            data= DatasetDict({"train": data[0].select(range(num_rows)) if num_rows else data[0],"test": data[1]})
            data['train']= data['train'].remove_columns(['start_second', 'end_second', 'local_path', 'speaker_id', 'normalized_text', 'index', 'id'])
            data['test']= data['test'].remove_columns(['start_second', 'end_second', 'local_path', 'speaker_id', 'normalized_text', 'index', 'id'])
            #data = [(x['audio'], x['sentiment']) for x in data]
        elif task == 'agegender':
            sp= ["train",'test']
            data=load_dataset("mozilla-foundation/common_voice_16_0", "en", split=sp, token= "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", trust_remote_code=True)
            data= DatasetDict({"train": data[0],"test": data[1]})
            data['train']= data['train'].remove_columns(['variant', 'segment', 'locale', 'accent', 'down_votes', 'up_votes', 'client_id', 'sentence'])
            data['test']= data['test'].remove_columns(['variant', 'segment', 'locale', 'accent', 'down_votes', 'up_votes', 'client_id', 'sentence'])
            # mask = ds.field('gender') != "" & ds.field('age') != ''
            # data = data.data.filter(~mask).to_table()
            data = data.filter(lambda rows: [g and a for g, a in zip(rows['gender'], rows['age']) ], batched=True)
            if num_rows:
                data['train']= data['train'].select(range(num_rows))
            #data = data.map(self.prepare_mozilla_dataset, desc="Preprocessing Mozilla Voice")
            data = data.cast_column("audio", Audio(sampling_rate=16000))
            
            #data = [(x['audio'], x['sentence'], x['age'], x['gender']) for x in data]
        else:
            raise ValueError("Task must be one of : [speechsum, noisedet, sentdet, agegender]")
        
        
        return data
        # ssum_ds = 
        # noise_ds = #, split=f"train.100[:{num_rows}]"
        # mozilla_voice = 
        # slue_ds = 

        # Preprocess Mozilla Voice
        

        # print("==!== Beginning Processing ==!==")
        # print(f'=!= Downsampling: Using min_size={num_rows} =!=')

        
        
        
        

        # cmumosi_texts = cmumosi_highlevel['cmumosi']['CMU_MOSI_TimestampedWords']['text']
        # cmumosi_texts = [entry[0] for entry in cmumosi_texts]
        # cmumosi_samples = [(entry['audio'], entry['sentiment']) for entry in cmumosi_texts]
        # ssum_samples = self.downsample_dataset(ssum_samples, min_size)
        # noise_samples = self.downsample_dataset(noise_samples, min_size)
        # mozilla_samples = self.downsample_dataset(mozilla_samples, min_size)
        # slue_samples = self.downsample_dataset(slue_samples, min_size)

        
        # noise_train, noise_test = train_test_split(noise_samples, test_size=0.2, random_state=42)
        # mozilla_train, mozilla_test = train_test_split(mozilla_samples, test_size=0.2, random_state=42)
        # slue_train, slue_test = train_test_split(slue_samples, test_size=0.2, random_state=42)

        # return {
        #     "summary": {
        #         "train": ssum_train,
        #         "test": ssum_test
        #     },
        #     "noise_prediction": {
        #         "train": noise_train,
        #         "test": noise_test
        #     },
        #     "age_gender_prediction": {
        #         "train": mozilla_train,
        #         "test": mozilla_test
        #     },
        #     "sentiment_analysis": {
        #         "train": slue_train,
        #         "test": slue_test
        #     }
        # }

        # Combine and split into train and test
        # combined_samples = ssum_samples + noise_samples + mozilla_samples + cmumosi_samples
        # combined_train, combined_test = train_test_split(combined_samples, test_size=0.2, random_state=42)

        # # Return the dataset with audio and metadata
        # return {
        #     "train": combined_train,
        #     "test": combined_test
        # }


if __name__ == '__main__':
    
    dp = DataProcessor()
    data= dp.load_and_process_data(task='agegender', num_rows=250)
    print(data)
    # data1 = dp.load_and_process_data(task='speechsum')
    # data2 = dp.load_and_process_data(task='noisedet')
    # data3 = dp.load_and_process_data(task='sentdet')
    # data4 = dp.load_and_process_data(task='agegender')
    # data=[data1, data2, data3, data4]
    # data[0]= data[0].map(prepare_dataset_speechsum)
    # data[1]= data[1].map(prepare_dataset_speechsum)
    # data[2]= data[2].map(prepare_dataset_speechsum)
    # #data[3]= data[3].map(prepare_dataset_speechsum)
    
    # for i in range(3):
    #     cols_to_remove = data[i]['train'].column_names
    #     print(cols_to_remove)
    #     cols_to_remove.remove("labels")
    #     cols_to_remove.remove("input_features")
    #     data[i] = data[i].remove_columns(cols_to_remove)
    #     print(data[i])
    
    # print('Bro you did it')
    # data_train = concatenate_datasets([data[0]['train'], data[1]['train'], data[2]['train']])
    # data_test = concatenate_datasets([data[0]['test'], data[1]['test'], data[2]['test']])
    # data= DatasetDict({"train": data_train,"test": data_test})
    # print(data)
    #data= data.train_test_split(test_size=0.3, seed=42)
    
    
    
    
    