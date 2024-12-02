in your transformers installation go to `/ocean/projects/cis240125p/<account_name>/<conda_installation path>/<encv_name>/lib/python/site-packages/transfomers/models/whisper/tokenization_whisper.py` and do the following changes.

1. Add `self.prompt= None` in line 324.
2. change the `prefix_tokens` method in line 406 to:  
```
        

        def prefix_tokens(self) -> List[int]:
        prev_token_id= self.convert_tokens_to_ids("<|startofprev|>")
        bos_token_id = self.convert_tokens_to_ids("<|startoftranscript|>")
        translate_token_id = self.convert_tokens_to_ids("<|translate|>")
        transcribe_token_id = self.convert_tokens_to_ids("<|transcribe|>")
        notimestamps_token_id = self.convert_tokens_to_ids("<|notimestamps|>")
        langs = tuple(LANGUAGES.keys())
        
        if self.language is not None:
            self.language = self.language.lower()
            if self.language in TO_LANGUAGE_CODE:
                language_id = TO_LANGUAGE_CODE[self.language]
            elif self.language in TO_LANGUAGE_CODE.values():
                language_id = self.language
            else:
                is_language_code = len(self.language) == 2
                raise ValueError(
                    f"Unsupported language: {self.language}. Language should be one of:"
                    f" {list(TO_LANGUAGE_CODE.values()) if is_language_code else list(TO_LANGUAGE_CODE.keys())}."
                )
        
        if self.task is not None:
            if self.task not in TASK_IDS:
                raise ValueError(f"Unsupported task: {self.task}. Task should be in: {TASK_IDS}")

        bos_sequence = [bos_token_id]
        if self.language is not None:
            bos_sequence.append(bos_token_id + 1 + langs.index(language_id))
        if self.task is not None:
            task_id= self.convert_tokens_to_ids(f"<|{self.task}|>")
            bos_sequence.append(transcribe_token_id if self.task == "transcribe" else task_id)
        if not self.predict_timestamps:
            bos_sequence.append(notimestamps_token_id)
        return bos_sequence  
```

3. change the `build_inputs_with_special_tokens` method in line 442 to:  

```
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        token_ids_1= self.prompt
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        start_of_prev_id = self.all_special_ids[-7]
        
        return [start_of_prev_id] + token_ids_1[len(self.prompt)-4 : -1] + self.prefix_tokens + token_ids_0 + [self.eos_token_id]
```

4. in line 207 change `TASK_IDS` variable to `TASK_IDS = ["translate", "transcribe", 'speechsum', 'noisedet', 'sentdet', 'agegender', 'startoflm']`

To run an experiment run `finetune.py` and pass `--method` and `--task` args `--prompt` is optional and is used to pass prompts to whisper in either method  
Example `python finetune.py --method token --task noisedet --prompt "true , false"(optional)`

--method: which method is used to finetune whisper \[token , prompt\]  
--task: which task to train on \[speechsum , noisedet , sentdet , agegender\]  
--prompt: training whisper with prompts \["add your own prompt"\]   
