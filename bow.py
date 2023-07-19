import torch
from collections import Counter

def make_word_dictionary(data, threshold: int):
    # maps each word in vocab to BoW idx
    # No UNK index specified here, because this could lead to problems if UNK key is used as an actual word
    word_counts = Counter(word for sentence, _ in data for word in sentence)
    valid_words = [word for word, count in word_counts.items() if count >= threshold]
    word_to_ix = {word: ix for ix, word in enumerate(valid_words)}
    
    return word_to_ix


def make_label_dictionary(data):
    # maps each label in the data to an index
    label_to_ix = {}
    for _, label in data:
        if label not in label_to_ix:
            label_to_ix[label] = len(label_to_ix)
    return label_to_ix


def make_bow_vector(sentence, word_to_ix):
    # maps a sentence to BOW vector
    vec = torch.zeros(len(word_to_ix)+1) # +1 for UNK index
    for word in sentence: # count vectorization
        ix = word_to_ix[word] if word in word_to_ix else len(word_to_ix) # use last index for UNK
        vec[ix] += 1
    return vec.view(1, -1)


def make_label_vector(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])