# ASR-TermProject

## Whisper.py
Fine-tuning code for whisper, it supports adding new tokens or using prompts and LM fine tuning.

## eval.py
Includes evaluations methods for the four tasks (one sequence + three classification).
- sequence output
    - word error rate
- classification output
    - called Llama2-7b-hf to standardize the output and count exact match


### Notes:
- Code requires some changes in the source code of the transformers package. (Will be discussed with the team after integrating everything)
### TODO
- [ ] test with the full code
- [ ] add chatGPT for classification output standardization
