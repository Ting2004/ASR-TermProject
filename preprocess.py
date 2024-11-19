from datasets import load_dataset, concatenate_datasets, Audio
from datasets import DatasetDict


# need to login for huggingface
out_common_voice = "common_voice_16_en_spkinfo"
out_noise = "librispeech_noise"



def process_agegender():

    ds_train = load_dataset("mozilla-foundation/common_voice_16_0", "en", split="train", trust_remote_code=True)
    ds_vali = load_dataset("mozilla-foundation/common_voice_16_0", "en", split="validation", trust_remote_code=True)
    ds_test = load_dataset("mozilla-foundation/common_voice_16_0", "en", split="test", trust_remote_code=True)

    ds_train = ds_train.remove_columns(["up_votes", "down_votes", "accent", "locale", "segment", "variant"])
    ds_train = ds_train.filter(lambda row: row["gender"] != "" and row["age"] != "")

    ds_vali = ds_vali.remove_columns(["up_votes", "down_votes", "accent", "locale", "segment", "variant"])
    ds_vali = ds_vali.filter(lambda row: row["gender"] != "" and row["age"] != "")

    ds_test = ds_test.remove_columns(["up_votes", "down_votes", "accent", "locale", "segment", "variant"])
    ds_test = ds_test.filter(lambda row: row["gender"] != "" and row["age"] != "")


    ds = DatasetDict({
        "train":ds_train,
        "validation": ds_vali,
        "test":ds_test
    })
    ds.push_to_hub(out_common_voice, private=False)



def process_noise():
    # TODO edit this to take ALL SPLITS into account
    # total size would be probably 200 hrs (clean and noisy)
    # need to reconsider what is clean and what is noisy

    # the number for split indicate how much noise there is
    # smaller means more noisy

    ds_pub_test = load_dataset("distil-whisper/librispeech_asr-noise", "test-pub-noise", split="0", trust_remote_code=True)
    ds_pub_vali = load_dataset("distil-whisper/librispeech_asr-noise", "validation-pub-noise", split="0", trust_remote_code=True)

    ds_clean_test = load_dataset("distil-whisper/librispeech_asr-noise", "test-pub-noise", split="40", trust_remote_code=True)
    ds_clean_vali = load_dataset("distil-whisper/librispeech_asr-noise", "validation-pub-noise", split="40", trust_remote_code=True)


    true_column = ["true"] * len(ds_pub_test)
    ds_pub_test = ds_pub_test.add_column("noise", true_column)

    true_column = ["true"] * len(ds_pub_vali)
    ds_pub_vali = ds_pub_vali.add_column("noise", true_column)

    false_column = ["false"] * len(ds_clean_test)
    ds_clean_vali = ds_pub_vali.add_column("noise", false_column)

    false_column = ["false"] * len(ds_clean_vali)
    ds_clean_vali = ds_pub_vali.add_column("noise", false_column)


    noisy = concatenate_datasets([ds_pub_test, ds_pub_vali])
    clean = concatenate_datasets([ds_clean_test, ds_clean_vali])

    # regenerate train test split
    noisy = noisy.train_test_split(test=0.3)
    clean = clean.train_test_split(test=0.3)

    test = concatenate_datasets([clean["test"], noisy["test"]])
    train = concatenate_datasets([clean["train"], noisy["train"]])
    ds = DatasetDict({
        "train": train,
        "test": test
    })
    ds.push_to_hub(out_noise, private=False)


process_agegender()
