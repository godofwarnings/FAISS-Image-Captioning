from nltk.translate import bleu_score

def accuracy(predict, real):
    """
    Calculate BLEU score accuracy.
    
    Args:
        predict: List of predicted captions
        real: List of actual descriptions
        
    Returns:
        float: BLEU accuracy score
    """
    accuracy = 0
    for i, pre in enumerate(predict):
        references = real[i]
        score = bleu_score.sentence_bleu(references, pre)
        accuracy += score
    return accuracy/len(predict)

def accuracy_v2(predict, real):
    """
    Enhanced BLEU score calculation with preprocessing.
    
    Args:
        predict: List of predicted captions
        real: List of actual descriptions
        
    Returns:
        float: Enhanced BLEU accuracy score
    """
    lower_n_split = lambda x: x.lower().split()

    accuracy = 0
    for i, pre in enumerate(predict):
        refs = real[i]
        score = bleu_score.sentence_bleu(
            list(map(lambda ref: lower_n_split(ref), refs)), 
            lower_n_split(pre), 
            weights=(0.5, 0.5, 0, 0)
        )
        accuracy += score
    return accuracy/len(predict)