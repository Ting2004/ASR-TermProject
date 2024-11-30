# The following use of BERTScore is referenced from the official repository of BERTScore
# Repo: https://github.com/Tiiiger/bert_score

from bert_score import score

# Function to calculate BERTScore (Can use Batch Processing)
def score_BERTScore (reference, prediction):

    # if reference is single string
    if isinstance(reference, str) :
        reference = [reference]

    # if prediction is single string
    if isinstance(prediction, str) :
        prediction = [prediction]

    # Calculate BERTScore
    P, R, F1 = score(prediction, reference, lang='en', verbose=False, rescale_with_baseline=True)
    return F1

if __name__ == "__main__":

    with open("evaluation/hyps.txt") as f:
        cands = [line.strip() for line in f]

    with open("evaluation/refs.txt") as f:
        refs = [line.strip() for line in f]
    
    print(score_BERTScore(cands, refs))