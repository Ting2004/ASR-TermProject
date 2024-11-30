# pip install nltk



from nltk.translate.bleu_score import sentence_bleu, corpus_bleu


def score_BLEU (reference, prediction):
    bleu_score = sentence_bleu([reference], prediction)
    # bleu score is a single float 
    return bleu_score


print(score_BLEU("the cat was found under the bed", "found a cat under the bed"))