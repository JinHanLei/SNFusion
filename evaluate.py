from rouge import Rouge


def single_text_rouge(text_a, text_b):
    rouge = Rouge()
    scores = rouge.get_scores(text_a, text_b)
    return scores
