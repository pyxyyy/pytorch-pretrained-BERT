# python build-pos.py <train_file> <model_file>
"""
builds a bigram / trigram hidden markov model for pos tagging
"""
import math
import re
import sys
import datetime
import numpy as np
import _pickle as pk
from collections import defaultdict


START_SYMBOL = "<s>"
END_SYMBOL = "</s>"


def parse_sent_bigram(sent):
    sent_tokens = []
    sent_tags = []
    data = sent.strip().split()
    for datum in data:
        datum_token, datum_tag = datum.rsplit('/', 1)
        sent_tokens.append(datum_token)
        sent_tags.append(datum_tag)
    sent_tokens = [START_SYMBOL] + sent_tokens + [END_SYMBOL]
    sent_tags = [START_SYMBOL] + sent_tags + [END_SYMBOL]
    return sent_tokens, sent_tags


def parse_sent_trigram(sent):
    sent_tokens = []
    sent_tags = []
    data = sent.strip().split()
    for datum in data:
        datum_token, datum_tag = datum.rsplit('/', 1)
        sent_tokens.append(datum_token)
        sent_tags.append(datum_tag)
    sent_tokens = [START_SYMBOL, START_SYMBOL] + sent_tokens + [END_SYMBOL]
    sent_tags = [START_SYMBOL, START_SYMBOL] + sent_tags + [END_SYMBOL]
    return sent_tokens, sent_tags


def get_transition_probs_bigram(file_tags):

    unigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)

    def add_sent_unigram_counts(sent_tag):
        for tag in sent_tag:
            unigram_counts[tag] += 1

    def add_sent_bigram_counts(sent_tag):
        prev_tag = sent_tag[0]
        for tag in sent_tag[1:]:
            bigram_counts[(prev_tag, tag)] += 1
            prev_tag = tag

    for sent_tag in file_tags:
        add_sent_unigram_counts(sent_tag)
        add_sent_bigram_counts(sent_tag)

    """
    kneser = defaultdict(int)
    kneser_discount = 0.75

    def get_kneser_lambda(tag):
        denom = unigram_counts[tag]
        numer = 0.0
        for (t1, t2) in bigram_counts:
            if t1 == tag:
                numer += 1
        numer = kneser_discount * numer
        return numer / denom

    def kneser_ney():

        context = defaultdict(int)
        for (t1, t2) in bigram_counts:
            context[t2] += 1

        for (t1, t2), count in bigram_counts.items():
            base = float(max(count - kneser_discount, 0)) / unigram_counts[t1]
            theta = get_kneser_lambda(t1)
            pcont = float(context[t2]) / len(bigram_counts)
            kneser[(t1, t2)] = base + (theta * pcont)
            print(kneser[(t1, t2)])

    kneser_ney()
    return kneser
    """

    unigram_probs = defaultdict(float)
    bigram_probs = defaultdict(float)

    def deleted_interpolation():
        # referenced Brants, 2000 (https://arxiv.org/pdf/cs/0003055.pdf) for the algorithm
        lambdas = [0.0, 0.0]  # [lambda_unigram, lambda_bigram]
        for (t2, t3), count in bigram_counts.items():
            if count > 0:
                denom = unigram_counts[t2] - 1
                case_bigram = 0 if denom == 0 else (bigram_counts[(t2, t3)] - 1) / denom
                denom = sum(unigram_counts.values()) - 1
                case_unigram = 0 if denom == 0 else (unigram_counts[t3] - 1) / denom
                argmax = np.argmax([case_bigram, case_unigram])
                if argmax == 1:
                    lambdas[0] += count
                elif argmax == 0:
                    lambdas[1] += count
        norm_factor = sum(lambdas)
        normed_lambdas = [_ / norm_factor for _ in lambdas]
        return normed_lambdas

    interpolation_weights = deleted_interpolation()
    weight_unigram = interpolation_weights[0]
    weight_bigram = interpolation_weights[1]

    def add_unigram_probs():
        denom = sum(unigram_counts.values())
        for unigram, unigram_count in unigram_counts.items():
            unigram_probs[unigram] = unigram_count / denom

    def add_bigram_probs():
        for bigram, bigram_count in bigram_counts.items():
            bigram_probs[bigram] = bigram_count / unigram_counts[bigram[0]]

    add_unigram_probs()
    add_bigram_probs()

    transition_probs = defaultdict(float)
    for t2 in unigram_probs:
        for t3 in unigram_probs:
            transition_probs[(t2, t3)] = weight_unigram * unigram_probs[t3] \
                                             + weight_bigram * bigram_probs[(t2, t3)]
            transition_probs[(t2, t3)] = math.log(transition_probs[(t2, t3)])

    return transition_probs


def get_transition_probs_trigram(file_tags):

    unigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    trigram_counts = defaultdict(int)

    def add_sent_unigram_counts(sent_tag):
        for tag in sent_tag:
            unigram_counts[tag] += 1

    def add_sent_bigram_counts(sent_tag):
        prev_tag = sent_tag[0]
        for tag in sent_tag[1:]:
            bigram_counts[(prev_tag, tag)] += 1
            prev_tag = tag

    def add_sent_trigram_counts(sent_tag):
        prev_tags = [sent_tag[0], sent_tag[1]]
        for tag in sent_tag[2:]:
            trigram_counts[(prev_tags[0], prev_tags[1], tag)] += 1
            prev_tags = [prev_tags[1], tag]

    for sent_tag in file_tags:
        add_sent_unigram_counts(sent_tag)
        add_sent_bigram_counts(sent_tag)
        add_sent_trigram_counts(sent_tag)

    unigram_probs = defaultdict(float)
    bigram_probs = defaultdict(float)
    trigram_probs = defaultdict(float)

    def deleted_interpolation():
        # referenced Brants, 2000 (https://arxiv.org/pdf/cs/0003055.pdf) for the algorithm
        lambdas = [0.0, 0.0, 0.0]  # [lambda_unigram, lambda_bigram, lambda_trigram]
        for (t1, t2, t3), count in trigram_counts.items():
            if count > 0:
                denom = bigram_counts[(t1, t2)] - 1
                case_trigram = 0 if denom == 0 else (count - 1) / denom
                denom = unigram_counts[t2] - 1
                case_bigram = 0 if denom == 0 else (bigram_counts[(t2, t3)] - 1) / denom
                denom = sum(unigram_counts.values()) - 1
                case_unigram = 0 if denom == 0 else (unigram_counts[t3] - 1) / denom
                argmax = np.argmax([case_trigram, case_bigram, case_unigram])
                if argmax == 2:
                    lambdas[0] += count
                elif argmax == 1:
                    lambdas[1] += count
                elif argmax == 0:
                    lambdas[2] += count
        norm_factor = sum(lambdas)
        normed_lambdas = [_ / norm_factor for _ in lambdas]
        return normed_lambdas

    interpolation_weights = deleted_interpolation()
    weight_unigram = interpolation_weights[0]
    weight_bigram = interpolation_weights[1]
    weight_trigram = interpolation_weights[2]

    def add_unigram_probs():
        denom = sum(unigram_counts.values())
        for unigram, unigram_count in unigram_counts.items():
            unigram_probs[unigram] = unigram_count / denom

    def add_bigram_probs():
        for bigram, bigram_count in bigram_counts.items():
            bigram_probs[bigram] = bigram_count / unigram_counts[bigram[0]]

    def add_trigram_probs():
        for trigram, trigram_count in trigram_counts.items():
            trigram_probs[trigram] = trigram_count / bigram_counts[(trigram[0], trigram[1])]

    add_unigram_probs()
    add_bigram_probs()
    add_trigram_probs()

    transition_probs = defaultdict(float)
    for t1 in unigram_probs:
        for t2 in unigram_probs:
            for t3 in unigram_probs:
                transition_probs[(t1, t2, t3)] = weight_unigram * unigram_probs[t3] \
                                                 + weight_bigram * bigram_probs[(t2, t3)] \
                                                 + weight_trigram * trigram_probs[(t1, t2, t3)]
                transition_probs[(t1, t2, t3)] = math.log(transition_probs[(t1, t2, t3)])

    return transition_probs


def morph(token):
    if re.search(r'\d', token):
        return "_CHAR"
    elif not re.search(r'\w', token):
        return "_PUNC"
    elif re.search(r'[A-Z]', token):
        return "_CAMEL"
    elif re.search(r'[A-Z]*', token):
        return "_CAPS"
    elif re.search(r'[0-9]*[.,/-]?[0-9]*', token):
        return "_NUM"
    elif re.search(r'[a-z]*(ate|ify|ise|ize|ing)\b', token):
        return "_VBISH"
    elif re.search(r'[a-z]*(able|ible|ese|ful|ic|ish|ive|less|ly|ous|y)\b', token):
        return "_JJISH"
    elif re.search(r"[a-z]*(age|ance|ence|dom|ee|or|hood|ism|ist|ty|ment|ness|ry|ies|ship|sion|"
                   r"tion|xion)[s]?\b", token):
        return "_NNISH"
    return "_RARE"


def reduce_vocab(file_tokens):

    TOKEN_FREQ_THRESHOLD = 3

    full_vocab = defaultdict(int)
    for sent_tokens in file_tokens:
        for token in sent_tokens:
            full_vocab[token] += 1

    reduced_vocab = defaultdict(int)
    for token, freq in full_vocab.items():
        if freq > TOKEN_FREQ_THRESHOLD:
            reduced_vocab[token] = freq
        else:
            reduced_vocab[morph(token)] += freq

    return reduced_vocab


def get_emission_probs(file_tokens, file_tags, reduced_vocab):

    emission_counts = defaultdict(float)
    tag_counts = defaultdict(int)

    for sent_token, sent_tag in zip(file_tokens, file_tags):
        for token, tag in zip(sent_token, sent_tag):
            new_token = token if token in reduced_vocab else morph(token)
            emission_counts[(tag, new_token)] += 1
            tag_counts[tag] += 1

    emission_probs = defaultdict(float)
    for tag_token, emission_count in emission_counts.items():
        emission_probs[tag_token] = emission_count / (tag_counts[tag_token[0]] + 1)
        emission_probs[tag_token] = math.log(emission_probs[tag_token])

    return emission_probs


def train_model(train_file, model_file):

    print("Training bigram model...")
    with open(train_file, 'r') as fin:

        # parse training data
        file_tokens = []
        file_tags = []
        for sent in fin:
            sent_tokens, sent_tags = parse_sent_bigram(sent)
            file_tokens.append(sent_tokens)
            file_tags.append(sent_tags)

        # get transition probabilities
        transition_probs = get_transition_probs_bigram(file_tags)

        # get emission probabilities
        reduced_vocab = reduce_vocab(file_tokens)
        emission_probs = get_emission_probs(file_tokens, file_tags, reduced_vocab)

        # get set of all tags
        full_tags = set()
        for sent_tags in file_tags:
            for tag in sent_tags:
                full_tags.add(tag)

    print("Saving bigram model...")
    with open(model_file, 'wb') as fout:
        pk.dump(reduced_vocab, fout)
        pk.dump(full_tags, fout)
        pk.dump(transition_probs, fout)
        pk.dump(emission_probs, fout)
    print('Done')


if __name__ == "__main__":
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    start_time = datetime.datetime.now()
    train_model(train_file, model_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
