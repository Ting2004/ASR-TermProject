from transformers import pipeline
from evaluate import load
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

LLM_MODEL = "meta-llama/Llama-2-7b-hf" # could we use ChatGPT-4o API instead?

noisedet = """The task is to determine whether the surrounding of the speech is noisy or not. The audio is recorded under no noise, white noise, or natural noise. Read the following output from the ASR model and report "True" for noisy and "False" for not noisy. For example,  "False" when the audio is clear or noise."""

sentiment_bank = ""
sentdet = f"""The task is to determine the sentiment of the speaker. The sentiment should be one of {sentiment_bank}. Read the following output from the ASR model and report one of the sentiment. For example. "happy" when the speaker shows joy."""

agegender = """The task is to provide the age and gender of the speaker. Read the following result from the ASR model and output in the     output of "<age>,  <gender>". For example "25, female"."""


TASK_PROMPTS = {
    '<|noisedet|>' :  noisedet,
    '<|sentdet|>'  :  sentdet,
    '<|agegender|>':  agegender
    }


class EvalMetric:
    def __init__(self, task_token):
        """
        Initialize EvalMetric with a task name.

        Args:
            task_name (str): The name of the task to determine the evaluation method.
            should be one of the followings: "<|speechsum|>", "<|noisedet|>", "<|sentdet|>", "<|agegender|>"
        """
        self.llm = None
        self.wer_metric = None
        self.task_token = task_token
        self.task_name = task_token[2:-2].lower()  # Normalize task name for consistency
        self.eval_method = self._select_evaluation_method()
       


    def _select_evaluation_method(self):
        """
        Selects the evaluation method based on the task name.

        Returns:
            str: The evaluation method ("WER" or "LLM-assisted").
        """
        sequence_tasks = ["speechsum"]
        classification_tasks = ["noisedet", "sentdet", "agegender"]
        if self.task_name in sequence_tasks:
            self.wer_metric = load("wer")
            return "WER"
        elif self.task_name in classification_tasks:
            self._initialize_llm()
            print("llm_loded")
            return "LLM-assisted"  
        else:
            raise ValueError
        

    def _initialize_llm(self):
        # if LLM_MODEL != "chatGPT":

        self.model = LlamaForCausalLM.from_pretrained(LLM_MODEL)
        self.tokenizer = LlamaTokenizer.from_pretrained(LLM_MODEL)
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device=0,
            )
        self.llm = pipe
        # else:
        #     # TODO add the pipeline for chatGPT API
        #     raise ValueError

    def evaluate(self, reference, hypothesis):
        """
        Perform the evaluation based on the selected method.

        Args:
            reference (str): The ground truth text.
            hypothesis (str): The system's output text.

        Returns:
            float or str: Evaluation score or LLM-assisted feedback.
        """
        if self.eval_method == "WER":
            return self._calculate_wer(reference, hypothesis)
        elif self.eval_method == "LLM-assisted":
            return self._llm_assisted_evaluation(reference, hypothesis)

    def _calculate_wer(self, reference, hypothesis):
        """
        Calculates the Word Error Rate (WER) using the Hugging Face evaluate library.

        Args:
            reference (str): The ground truth text.
            hypothesis (str): The system's output text.

        Returns:
            float: The WER score as a percentage.
        """
        wer_score = self.wer_metric.compute(references=[reference], predictions=[hypothesis])
        return (reference, hypothesis, wer_score * 100)  # Return WER as a percentage

    def _llm_assisted_evaluation(self, reference, hypothesis):
        """
        Placeholder for LLM-assisted evaluation.

        Args:
            reference (str): The ground truth text.
            hypothesis (str): The system's output text.

        Returns:
            str: LLM-assisted feedback or evaluation result.
        """
        setting = """You are a post-processor for a automatic speech recognition output.
            Follow the task instruction and output format below, show your thought process to the final decision step-by-step."""
        instruction = TASK_PROMPTS[self.task_token]
        prompt = f"""
            {setting}
            {instruction}
            The output from the ASR model: "{hypothesis}". Report your final decision is quotation marks.
            """
        sequences = self.llm(
            prompt.strip(),
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=100,
            )
        for seq in sequences:
            print(f"{seq['generated_text']}")
        standardized_output = sequences[0]['generated_text'].strip()
        return (reference, standardized_output, int(reference == standardized_output))
        


if __name__ == "__main__":
    metric = EvalMetric("<|agegender|>")

    # Example references and hypotheses
    references = [
        "20, female",
        "70, male",
        "25, male",
        "15, male",
        "42, female"]
    hypothesis = [
        "This might be a girl speaking in a high pitch",
        "The old is talking to his descents",
        "20s, man",
        "teenager, boy",
        "42 year old mom"]

    for r, h in zip(references, hypothesis):
        ref, hyp, res = metric.evaluate(r, h)
        print(f"{ref}|{hyp}|{res}")

    metric = EvalMetric("<|speechsum|>")
    references = [
        "The speaker is talking about their study experience at college"]
    hypothesis = [
        "The college student talks about their busy daily life at college. Poor kid."]
    for r, h in zip(references, hypothesis):
        ref, hyp, res = metric.evaluate(r, h)
        print(f"{ref}|{hyp}|{res}")