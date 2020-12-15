#!/usr/local/bin/python3

"""
Computes a "responsivity" measure, the amount of semantic overlap between
successive speaker turns in a transcribed conversation.
Semantic overlap is measured using cosine distance between vectors
produced by Sentence-BERT for pairs of sentences taken 
near the boundary of the speaker turn.

Outputs a JSON description of the speaker turns.

Outputs an HTML file to out.html that highlights the responsive (and responded-to) sentences
highlighted in red and orange, respectively.

The input must be in LVN snippets format:

[ {"audio_start_offset": 0.0, "audio_end_offset": 10, "speaker_id": "123", "content": "Hello there."},
  ... ]


Example:
   aws s3 cp  s3://cortico-adhoc-data/leaven/snippets2/70.json - | ./generate_responsivity_stats.py
"""

import json
import sys

import numpy as np
from scipy.spatial import distance
import spacy
from sentence_transformers import SentenceTransformer, util

import utils.lexical_metrics as lexical_metrics


# A window is a sequence of sentences that occurs either at the beginning
# or the end of a speaker turn.  This constant is the number of sentences to
# use for the responsivity metric
SENTENCE_WINDOW = 3

# Ignore sentences with fewer than this many words
MIN_WORDS = 6

# Sentence B is considered responsive to sentence A if its cosine
# similarity is greater than this threshold.
MIN_COS_SIMILARITY_FOR_RESPONSIVENESS = 0.55

def init_spacy():
    return spacy.load("en_core_web_sm")

def init_sentence_encoder():
    return SentenceTransformer('bert-base-nli-mean-tokens')

def measure_responsivity_for_window_pair(window1, window2, encoder):
    embeddings1 = [get_embedding(s, encoder) for s in window1]
    embeddings2 = [get_embedding(s, encoder) for s in window2]
    max_responsivity = 0.0
    best_pair = [0, 0]
    for i, emb1 in enumerate(embeddings1):
        sentence1 = window1[i]
        if len(sentence1) < MIN_WORDS:
            continue
        for j, emb2 in enumerate(embeddings2):
            sentence2 = window2[j]
            if len(sentence2) < MIN_WORDS:
                continue
            resp = measure_responsivity_for_sentence_pair(emb1, emb2)
            if resp > max_responsivity:
                max_responsivity = resp
                best_pair = [i, j]
    return {"max_responsivity": max_responsivity, "best_pair": best_pair}

def cos_similarity(vA, vB):
    return 1.0 - distance.cosine(vA, vB)

def measure_responsivity_for_sentence_pair(embedding1, embedding2):
    return cos_similarity(embedding1, embedding2)

def get_embedding(sentence, encoder):
    embedding = encoder.encode([str(sentence)], convert_to_tensor=False)
    return np.array(embedding).tolist()

def responsivity_metrics(snippets, nlp, encoder):
    last_speaker_turn_snippets = None
    results = []

    for i, speaker_turn_snippets in enumerate(lexical_metrics.generate_speaker_turns(snippets)):
        # This is a transition between speakers.  Extract the last few sentences
        # of the previous speaker, and the first few sentences of the current speaker.
        current_speaker_content, _ = lexical_metrics.snippets_to_content_string(speaker_turn_snippets)
        if last_speaker_turn_snippets:
            last_speaker_content, _ = lexical_metrics.snippets_to_content_string(last_speaker_turn_snippets)
            all_sentences1 = list(nlp(last_speaker_content).sents)
        else:
            all_sentences1 = [""]
        all_sentences2 = list(nlp(current_speaker_content).sents)
        window_sentences1 = all_sentences1[-SENTENCE_WINDOW:]
        window_sentences2 = all_sentences2[:SENTENCE_WINDOW]
        responsivity = measure_responsivity_for_window_pair(window_sentences1,
                                                            window_sentences2, encoder)

        res = {"speaker_id": speaker_turn_snippets[0]["speaker_id"],
               "sentences": [str(s) for s in all_sentences2],
               "window_sentences_A": [str(s) for s in window_sentences1],
               "window_sentences_B": [str(s) for s in window_sentences2],
               "sentences_A": [str(s) for s in all_sentences1],
               "best_pairing": responsivity["best_pair"],
               "responsivity": responsivity["max_responsivity"],
               "speaker_turn_idx": i}
        res["speaker_name"] = speaker_turn_snippets[0].get("speaker_name", res["speaker_id"])
        results.append(res)
        last_speaker_turn_snippets = speaker_turn_snippets

    sentence_classes = {"responded_sentence": 0,
                        "response_sentence": 0,
                        "neutral_sentence": 0}
    response_speaker_turns = set()
    for i, res in enumerate(results):
        for j, s in enumerate(res["sentences"]):
            # Is this responded to by the next speaker?
            if ((i < len(results) - 1)
                and (results[i+1]["responsivity"] >= MIN_COS_SIMILARITY_FOR_RESPONSIVENESS)
                and (results[i+1]["best_pairing"][0] == j - len(res["sentences"]) +
                     SENTENCE_WINDOW)):
                sentence_classes["responded_sentence"] += 1
            # Is this a response to the previous speaker?
            elif (res["responsivity"] >= MIN_COS_SIMILARITY_FOR_RESPONSIVENESS
                  and res["best_pairing"][1] == j):
                sentence_classes["response_sentence"] += 1
                response_speaker_turns.add(res["speaker_turn_idx"])
            else:
                sentence_classes["neutral_sentence"] += 1

    return results, sentence_classes, response_speaker_turns


if __name__ == "__main__":
    json_file = sys.argv[1]
    with open(json_file, 'r') as f:
        snippets = json.load(f)
    f.close()

    nlp = init_spacy()
    encoder = init_sentence_encoder()
    print("spaCy and sentence-BERT initialized.")

    results, sentence_classes, response_speaker_turns = responsivity_metrics(snippets, nlp, encoder)
    print(results)
    print(sentence_classes)
    print(response_speaker_turns)

