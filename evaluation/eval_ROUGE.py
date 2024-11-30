# pip install rouge-score
from rouge_score import rouge_scorer



# rouge1: unigram
# rouge2: bigram
# rougeL: longest common subsequence
# rougeS: skipgram
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def score_ROUGE (reference, prediction):
    scores = scorer.score(reference, prediction)
    # {"rouge1": Score(precision=0.0, recall=0.0, fmeasure=0.0),
    #  "rouge2": Score(precision=0.0, recall=0.0, fmeasure=0.0)}
    return scores


print(score_ROUGE("the cat was found under the bed", "found a cat under the bed"))