# python run-pos.py <test_file> <model_file> <output_file>
"""
runs a trained bigram / trigram hidden markov model for pos tagging
"""

import re
import sys
import datetime
import _pickle as pk
from collections import defaultdict
from operator import itemgetter

START_SYMBOL = "<s>"
END_SYMBOL = "</s>"


def load_model(model_file):
    with open(model_file, 'rb') as fin:
        vset = pk.load(fin)
        tset = pk.load(fin)
        transition_probs = pk.load(fin)
        emission_probs = pk.load(fin)
    return vset, tset, transition_probs, emission_probs


def morph(token):
    if re.search(r'\d', token):
        return "_CHAR"
    elif not re.search(r'\w', token):
        return "_PUNC"
    elif re.search(r'([A-Za-z]+[-][A-Za-z]+)', token):
        return "_HYPH"
    elif re.search(r'([A-Z][a-z]+)', token):
        return "_CAMEL"
    elif re.search(r'[A-Z]+', token):
        return "_CAPS"
    elif re.search(r'(?:\d+\.)?\d+,\d+', token):
        return "_NUM"
    elif re.search(r'[a-z]*(ate|ify|ise|ize|ing)\b', token):
        return "_VBISH"
    elif re.search(r'[a-z]*(able|ible|ese|ful|ic|ish|ive|less|ly|ous|y)\b', token):
        return "_JJISH"
    elif re.search(r"[a-z]*(age|ance|ence|dom|ee|or|hood|ism|ist|ty|ment|ness|ry|ies|ship|sion|"
                   r"tion|xion)[s]?\b", token):
        return "_NNISH"
    return "_RARE"


def bigram_viterbi(sent_tokens, vset, tset, transition_probs, emission_probs):
    NEGATIVE_INF = float('-inf')

    bp = {}
    pi = defaultdict(float)
    pi[(0, START_SYMBOL)] = 1

    tokens = [morph(sent_token) if sent_token not in vset else sent_token for sent_token in sent_tokens]
    for k in range(1, len(tokens) + 1):
        for u in tset:
            max_pi = NEGATIVE_INF
            max_bp = None
            for prev_tag in tset:
                pi_val = pi[(k - 1, prev_tag)] + transition_probs.get((prev_tag, u), -1000) + emission_probs.get(
                    (prev_tag, tokens[k - 1]), -1000)
                if pi_val > max_pi:
                    max_pi = pi_val
                    max_bp = prev_tag
            pi[(k, u)] = max_pi
            bp[(k, u)] = max_bp

    max_pi = NEGATIVE_INF
    max_u = None
    for u in tset:
        pi_val = pi[(len(tokens) - 1, u)] + transition_probs.get((u, END_SYMBOL), -1000)
        if pi_val > max_pi:
            max_pi = pi_val
            max_u = u

    sent_tags = [None for _ in sent_tokens]
    n = len(sent_tags)
    sent_tags[-1] = max_u
    for k in range(n - 2, -1, -1):
        sent_tags[k] = bp[(k + 1, sent_tags[k + 1])]

    tagged_sent = ""
    for i in range(len(sent_tokens)):
        tagged_sent = tagged_sent + sent_tokens[i] + '/' + sent_tags[i] + ' '
    tagged_sent = tagged_sent[:-1]

    return tagged_sent


def trigram_beam_viterbi(sent_tokens, vset, tset, transition_probs, emission_probs):

    NEGATIVE_INF = float('-inf')
    BEAM_THRESHOLD = 3

    bp = {}
    pi = defaultdict(float)
    pi[(0, START_SYMBOL, START_SYMBOL)] = 1

    tokens = [morph(sent_token) if sent_token not in vset else sent_token for sent_token in sent_tokens]
    for k in range(1, len(tokens) + 1):
        for u in tset:
            top_ws = [_ for _ in tset]
            for v in tset:
                max_pi = NEGATIVE_INF
                max_bp = None
                w_pis = []
                for w in top_ws:
                    w_pi = pi[(k - 1, w, u)] + transition_probs.get((w, u, v), -1000) \
                           + emission_probs.get((v, tokens[k - 1]), -1000)
                    w_pis.append((w, w_pi))
                    if w_pi > max_pi:
                        max_pi = w_pi
                        max_bp = w
                pi[(k, u, v)] = max_pi
                bp[(k, u, v)] = max_bp
                top_k_ws = sorted(w_pis, reverse=True, key=itemgetter(1))[0:BEAM_THRESHOLD]
                top_ws = [_[0] for _ in top_k_ws]

    max_pi = NEGATIVE_INF
    max_u_v = (None, None)
    for u in tset:
        for v in tset:
            u_v_pi = pi[(len(tokens), u, v)] + transition_probs.get((u, v, END_SYMBOL), -1000)
            if u_v_pi > max_pi:
                max_pi = u_v_pi
                max_u_v = (u, v)

    sent_tags = [None for _ in sent_tokens]
    n = len(sent_tags)
    sent_tags[n - 1] = max_u_v[1]
    sent_tags[n - 2] = max_u_v[0]
    for k in range(len(tokens) - 3, -1, -1):
        sent_tags[k] = bp[(k + 3, sent_tags[k + 1], sent_tags[k + 2])]

    tagged_sent = ""
    for i in range(len(tokens)):
        tagged_sent = tagged_sent + tokens[i] + '/' + sent_tags[i] + ' '
    tagged_sent = tagged_sent[:-1]

    return tagged_sent


def tag_sentence(test_file, model_file, out_file):
    print("Loading model...")
    vset, tset, transition_probs, emission_probs = load_model(model_file)

    print("Tagging sentences...")
    tagged_sents = []
    with open(test_file, 'r') as fin:
        for sent in fin:
            sent_tokens = sent.strip().split()
            tagged_sent = bigram_viterbi(sent_tokens, vset, tset, transition_probs, emission_probs)
            tagged_sents.append(tagged_sent)

    print("Writing results...")
    with open(out_file, 'w') as fout:
        fout.write('\n'.join(tagged_sents))

    print("Done")


if __name__ == "__main__":
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    start_time = datetime.datetime.now()
    tag_sentence(test_file, model_file, out_file)
    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
