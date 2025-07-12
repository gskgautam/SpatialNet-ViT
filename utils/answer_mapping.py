from collections import defaultdict

def build_answer_mappings(answers):
    """
    Build answer2idx and idx2answer mappings from a list of answers.
    Returns: (answer2idx, idx2answer)
    """
    unique_answers = sorted(set(answers))
    answer2idx = {ans: i for i, ans in enumerate(unique_answers)}
    idx2answer = {i: ans for ans, i in answer2idx.items()}
    return answer2idx, idx2answer

def get_answer2idx(answers):
    """
    Get answer2idx mapping from a list of answers.
    """
    return build_answer_mappings(answers)[0]

def get_idx2answer(answers):
    """
    Get idx2answer mapping from a list of answers.
    """
    return build_answer_mappings(answers)[1] 