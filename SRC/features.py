import re

def get_basic_features(text):
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]

    word_count = len(words)
    sentence_count = len(sentences)

    avg_sentence_length = 0
    if sentence_count > 0:
        avg_sentence_length = word_count / sentence_count

    unique_words = len(set(word.lower() for word in words))
    lexical_diversity = 0
    if word_count > 0:
        lexical_diversity = unique_words / word_count

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "lexical_diversity": round(lexical_diversity, 2)
    }