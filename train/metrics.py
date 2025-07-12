import numpy as np
import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import single_meteor_score

nltk.download('punkt', quiet=True)

def compute_bleu(references, hypotheses):
    """
    Compute BLEU score for a list of references and hypotheses.
    references: list of list of tokens
    hypotheses: list of tokens
    """
    return corpus_bleu([[ref] for ref in references], hypotheses)

def compute_meteor(references, hypotheses):
    """
    Compute METEOR score for a list of references and hypotheses.
    references: list of strings
    hypotheses: list of strings
    """
    scores = [single_meteor_score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
    return np.mean(scores)

def compute_rouge(references, hypotheses):
    """
    Compute ROUGE-L score for a list of references and hypotheses.
    references: list of strings
    hypotheses: list of strings
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, hyp)['rougeL'].fmeasure for ref, hyp in zip(references, hypotheses)]
    return np.mean(scores)

def compute_accuracy(preds, labels):
    """
    Compute accuracy for classification tasks.
    preds: list or np.array
    labels: list or np.array
    """
    return (np.array(preds) == np.array(labels)).mean()

def compute_mae(preds, targets):
    """
    Compute Mean Absolute Error (MAE) for regression/counting tasks.
    preds: list or np.array
    targets: list or np.array
    """
    return np.mean(np.abs(np.array(preds) - np.array(targets))) 