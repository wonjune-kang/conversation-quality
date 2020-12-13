#!/usr/local/bin/python3

"""
Computes various quantitative metrics from the text and metadata
of transcribed conversations between multiple speakers.

The input must be in LVN snippets format:

[ {"audio_start_offset": 0.0, "audio_end_offset": 10, "speaker_id": "123", "content": "Hello there."},
  ... ]

Example:
   aws s3 cp  s3://cortico-adhoc-data/leaven/snippets2/64.json - | ./metrics.py

"""

from textatistic import Textatistic
import json
import math
import sys

# Window for MATTR metric (see below).
_MATTR_WINDOW_SIZE = 1000


def mattr_metric(word_list):
    """Moving-Average Type Token Ratio measures lexical diversity.  See:
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.248.5206&rep=rep1&type=pdf """
    trailing_words = {}  # word -> count
    mattr_numer = 0
    mattr_denom = 0
    for i, w in enumerate(word_list):
        trailing_words[w] = trailing_words.get(w, 0) + 1
        if i >= _MATTR_WINDOW_SIZE:
            old_w = word_list[i - _MATTR_WINDOW_SIZE]
            trailing_words[old_w] -= 1  # must exist
            if not trailing_words[old_w]:
                del trailing_words[old_w]
            mattr_numer += len(trailing_words)
            mattr_denom += 1
    if mattr_denom:
        return mattr_numer / mattr_denom
    return None


def turn_taking_balance_metric(speaker_to_duration_sum_map):
    """Turn-taking balance is defined as the entropy of the speaker duration distribution
    divided by the maximum entropy for the same number of speakers."""
    duration_sum = sum(speaker_to_duration_sum_map.values())
    num_speakers = len(speaker_to_duration_sum_map)
    if num_speakers <= 1:
        return 1.0
    h = 0.0
    for x in speaker_to_duration_sum_map.values():
        if x > 0.0:
            h -= x / duration_sum * math.log(x / duration_sum)
    maxh = num_speakers * -1 / num_speakers * math.log(1 / num_speakers)
    return h / maxh

def snippets_to_content_string(snippets):
    is_crosstalk = False
    content_string = " ".join(x["content"] for x in snippets)
    if "[" in content_string:
        content_string = content_string[:content_string.index("[")] + content_string[content_string.index("]")+1:]
        is_crosstalk = True
    return content_string, is_crosstalk

def get_base_metrics(snippets):
    measures = get_measures(snippets)
    if measures["speaker_transitions"]:
        mean_speaker_silence = (
            measures["inter_speaker_silence"] / measures["speaker_transitions"]
        )
        interruption_rate = (
            measures["num_interruptions"] / measures["speaker_transitions"]
        )
    else:
        mean_speaker_silence = 0
        interruption_rate = 0
    if measures["num_words"]:
        mean_word_length = measures["word_length_sum"] / measures["num_words"]
    else:
        mean_word_length = 0.0
    if measures["duration_sum"]:
        words_per_hour = 3600 * measures["num_words"] / measures["duration_sum"]
        transitions_per_hour = (
            3600 * measures["speaker_transitions"] / measures["duration_sum"]
        )
    else:
        words_per_hour = 0.0
        transitions_per_hour = 0.0
    return [
        ("Words per hour", round(words_per_hour, 2)),
        (
            "Speech per turn",
            round(measures["duration_sum"] / (measures["speaker_transitions"] + 1), 2),
        ),
        ("Mean inter-speaker silence", round(mean_speaker_silence, 2)),
        ("Turn taking balance", round(measures.get("turn_taking_balance", 0.0), 3)),
        ("Interruption rate", round(interruption_rate, 2)),
        ("Grade level", round(measures.get("grade_level", 0), 2)),
        ("MATTR lexical diversity", round(measures.get("mattr_score", 0))),
        ("Mean word length", round(mean_word_length, 2)),
    ]


def generate_speaker_turns(snippets):
    """"Given a list of snippets, group them by speaker turn."""
    turn_snippets = []
    last_speaker_id = None
    for x in snippets:
        speaker_id = x["speaker_id"]
        if (last_speaker_id and speaker_id and speaker_id != last_speaker_id):
            yield turn_snippets
            turn_snippets = []
        turn_snippets.append(x)
        last_speaker_id = speaker_id
    if turn_snippets:
        # Last speaker turn
        yield turn_snippets

def get_measures(snippets):
    """Given a list of snippets for a conversation, return a dictionary of metrics."""
    duration_sum = 0.0
    word_length_sum = 0
    last_end_time = 0.0
    inter_speaker_silence = 0.0
    num_speaker_transitions = 0
    num_interruptions = 0
    word_count = 0

    all_content = []
    all_words = []
    speaker_to_duration_sum_map = {}  # speaker_id -> seconds

    # Split data into speaker turns and accumulate stats for each turn
    for speaker_turn_snippets in generate_speaker_turns(snippets):
        speaker_duration = sum(x["audio_end_offset"] - x["audio_start_offset"] for x in speaker_turn_snippets)
        speaker_id = speaker_turn_snippets[0]["speaker_id"]
        speaker_to_duration_sum_map[speaker_id] = (
            speaker_to_duration_sum_map.get(speaker_id, 0.0) + speaker_duration
        )
        speaker_content, is_crosstalk = snippets_to_content_string(speaker_turn_snippets)
        words = speaker_content.split()
        all_words += words
        duration_sum += speaker_duration
        word_count += len(words)
        word_length_sum += len(speaker_content) - len(words) + 1

        inter_speaker_gap = speaker_turn_snippets[0]["audio_start_offset"] - last_end_time
        if not last_end_time or inter_speaker_gap < 0 or inter_speaker_gap > 20:
            # Beginning of clip or very long pause.  Ignore as transition
            pass
        else:
            inter_speaker_silence += inter_speaker_gap
            num_speaker_transitions += 1
            if inter_speaker_gap < 0.0001:
                num_interruptions += 1
            elif is_crosstalk:
                num_interruptions += 1
        last_end_time = speaker_turn_snippets[-1]["audio_end_offset"]
        all_content.append(speaker_content)

    if word_count:
        grade_level = Textatistic(" ".join(all_content)).fleschkincaid_score
    else:
        grade_level = 0.0
    mattr_score = mattr_metric(all_words)

    x = {
        "duration_sum": duration_sum,
        "num_snippets": len(snippets),
        "num_words": word_count,
        "num_interruptions": num_interruptions,
        "word_length_sum": word_length_sum,
        "grade_level": grade_level,
        "turn_taking_balance": turn_taking_balance_metric(speaker_to_duration_sum_map),
        "num_speakers": len(speaker_to_duration_sum_map),
        "inter_speaker_silence": inter_speaker_silence,
        "speaker_transitions": num_speaker_transitions
    }
    if mattr_score:
        x["mattr_score"] = mattr_score

    return x


if __name__ == "__main__":
    json_file = sys.argv[1]
    with open(json_file, 'r') as f:
        snippets = json.load(f)
    f.close()

    base_metrics = get_base_metrics(snippets)
    print(base_metrics)
            

