from datasets import load_dataset, DatasetDict
from mmsdk import mmdatasdk
from sklearn.model_selection import train_test_split
import numpy as np
import random

token = "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"

class DataProcessor:
    def prepare_mozilla_dataset(batch):
        """Function to preprocess the Mozilla Common Voice dataset"""
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

    def load_and_process_data(self):
        ssum_ds = load_dataset("komats/mega-ssum")
        noise_ds = load_dataset("distil-whisper/librispeech_asr-noise", "test-pub-noise")
        mozilla_voice = load_dataset("mozilla-foundation/common_voice_17_0", "en", token=token, trust_remote_code=True) # , split="train[:10%]")
        cmumosi_highlevel = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel, 'cmumosi/')

        mozilla_voice = mozilla_voice.map(self.prepare_mozilla_dataset, desc="Preprocessing Mozilla Voice")

        ssum_texts = ssum_ds['train']['summary']
        noise_texts = noise_ds['test']['text']
        mozilla_texts = mozilla_voice['train']['sentence']

        cmumosi_texts = cmumosi_highlevel['cmumosi']['CMU_MOSI_TimestampedWords']['text']
        cmumosi_texts = [entry[0] for entry in cmumosi_texts]

        min_size = min(len(ssum_texts), len(noise_texts), len(mozilla_texts), len(cmumosi_texts))
        ssum_texts = self.downsample_dataset(ssum_ds['train'], min_size)
        noise_texts = self.downsample_dataset(noise_ds['test'], min_size)
        mozilla_texts = self.downsample_dataset(mozilla_voice['train'], min_size)
        cmumosi_texts = self.downsample_dataset(cmumosi_highlevel['cmumosi']['CMU_MOSI_TimestampedWords'], min_size)

        combined_texts = ssum_texts + noise_texts + mozilla_texts + cmumosi_texts
        combined_train, combined_test = train_test_split(combined_texts, test_size=0.2, random_state=42)

        return {
            "train": combined_train,
            "test": combined_test
        }

if __name__ == '__main__':
    dp = DataProcessor()
    data = dp.load_and_process_data()
    print(f"Train size: {len(data['train'])}, Test size: {len(data['test'])}")
