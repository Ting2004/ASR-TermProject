from datasets import load_dataset, DatasetDict, concatenate_datasets
from mmsdk import mmdatasdk
from sklearn.model_selection import train_test_split
import numpy as np
import random
import os

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

    def load_and_process_data(self, num_rows=1000):
        # Load datasets
        ssum_ds = load_dataset("komats/mega-ssum", split=f"core[:{num_rows}]")
        noise_clean_ds = load_dataset("distil-whisper/librispeech_asr-noise", "test-pub-noise", split="40")
        noise_clean_ds = noise_clean_ds.add_column("noise", ["false"] * len(noise_clean_ds))
        noise_noisy_ds = load_dataset("distil-whisper/librispeech_asr-noise", "test-pub-noise", split="0")
        noise_noisy_ds = noise_noisy_ds.add_column("noise", ["true"] * len(noise_noisy_ds))
        noise_ds = concatenate_datasets([noise_clean_ds, noise_noisy_ds])



        mozilla_voice = load_dataset("mozilla-foundation/common_voice_5_1", "en", split=f"train[:{num_rows}]", token=token)
        slue_ds = load_dataset("asapp/slue", "voxceleb", split=f"train[:{num_rows}]", token=token)

        # Preprocess Mozilla Voice
        mozilla_voice = mozilla_voice.filter(lambda row: row["gender"] != "" and row["age"] != "")
        mozilla_voice = mozilla_voice.map(self.prepare_mozilla_dataset, desc="Preprocessing Mozilla Voice")

        print("==!== Beginning Processing ==!==")
        print(f'=!= Downsampling: Using min_size={num_rows} =!=')

        ssum_samples = [(x['audio'], x['summary']) for x in ssum_ds]
        noise_samples = [(x['audio'], x['noise']) for x in noise_ds]
        mozilla_samples = [(x['audio'], x['sentence'], x['age'], x['gender']) for x in mozilla_voice]
        slue_samples = [(x['audio'], x['sentiment']) for x in slue_ds]

        # cmumosi_texts = cmumosi_highlevel['cmumosi']['CMU_MOSI_TimestampedWords']['text']
        # cmumosi_texts = [entry[0] for entry in cmumosi_texts]
        # cmumosi_samples = [(entry['audio'], entry['sentiment']) for entry in cmumosi_texts]
        # ssum_samples = self.downsample_dataset(ssum_samples, min_size)
        # noise_samples = self.downsample_dataset(noise_samples, min_size)
        # mozilla_samples = self.downsample_dataset(mozilla_samples, min_size)
        # slue_samples = self.downsample_dataset(slue_samples, min_size)

        ssum_train, ssum_test = train_test_split(ssum_samples, test_size=0.2, random_state=42)
        noise_train, noise_test = train_test_split(noise_samples, test_size=0.2, random_state=42)
        mozilla_train, mozilla_test = train_test_split(mozilla_samples, test_size=0.2, random_state=42)
        slue_train, slue_test = train_test_split(slue_samples, test_size=0.2, random_state=42)

        return {
            "summary": {
                "train": ssum_train,
                "test": ssum_test
            },
            "noise_prediction": {
                "train": noise_train,
                "test": noise_test
            },
            "age_gender_prediction": {
                "train": mozilla_train,
                "test": mozilla_test
            },
            "sentiment_analysis": {
                "train": slue_train,
                "test": slue_test
            }
        }

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
    data = dp.load_and_process_data()
    print(data.keys())
