#!/usr/bin/env python3

"""
Converts LVN conversation transcripts to LVN snippets format (JSON).

The output will have the following format:

[ {"audio_start_offset": 0.0, "audio_end_offset": 10, "speaker_id": "123", "content": "Hello there."},
  ... ]
"""

import os
import json
import pandas as pd


def get_transcript_from_id(_audio_id, conversations):

    def get_only_words(triplets):
        return [t[0] for t in triplets]

    diarization = conversations[_audio_id]['diarization']
    words = conversations[_audio_id]['words']

    speaker_turns = []
    prev_turn = 0
    for turn in diarization:
        spkr_label = turn['spkr_label']
        audio_start_offset = turn['start_seconds']
        audio_end_offset = turn['end_seconds']
        turn_words = get_only_words(words[prev_turn:turn['end_word_idx']+1])
        content = " ".join(turn_words)
        
        speaker_turn = {}
        speaker_turn['audio_start_offset'] = audio_start_offset
        speaker_turn['audio_end_offset'] = audio_end_offset
        speaker_turn['speaker_id'] = spkr_label
        speaker_turn['content'] = content
        speaker_turns.append(speaker_turn)

        prev_turn = turn['end_word_idx']+1

    return speaker_turns


if __name__ == '__main__':
    lvn_collections = pd.read_csv('../data/lvn/collections.csv')
    lvn_conversations_index = pd.read_csv('../data/lvn/conversations_index.csv')
    
    with open('../data/lvn/conversations.json') as f:
        lvn_conversations = json.loads(f.read())
    
    lvn_covid_convs = lvn_conversations_index[lvn_conversations_index.collection_id == 21]
    lvn_test_convs = lvn_covid_convs[lvn_covid_convs.id < 455]

    json_save_dir = "../data/json/lvn_json2"
    os.makedirs(json_save_dir, exist_ok=True)

    for _, lvn_conv in lvn_test_convs.iterrows():
        conv_id = int(lvn_conv.id)
        audio_id = lvn_conv._audio_id
        speaker_turns = get_transcript_from_id(audio_id, lvn_conversations)
        
        path_to_json = os.path.join(json_save_dir, str(conv_id)+".json")
        with open(path_to_json, 'w') as outfile:
            json.dump(speaker_turns, outfile)
        outfile.close()


