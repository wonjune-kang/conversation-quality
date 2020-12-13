#!/usr/bin/env python3

"""
Converts conversation transcripts in either Zoom VTT or Rev txt format
to LVN snippets format (JSON).

The output will have the following format:

[ {"audio_start_offset": 0.0, "audio_end_offset": 10, "speaker_id": "123", "content": "Hello there."},
  ... ]
"""

import os
import json
from collections import OrderedDict


def parse_speaker_turn_vtt(line):
    return int(line)

def parse_timestamp_vtt(line):
    def timestamp_to_secs(timestamp):
        hour, minute, seconds = timestamp.split(":")
        return round(3600*float(hour) + 60*float(minute) + float(seconds), 3)

    split_line = line.split()
    audio_start_offset = timestamp_to_secs(split_line[0])
    audio_end_offset = timestamp_to_secs(split_line[2])
    return audio_start_offset, audio_end_offset

def parse_content_vtt(line):
    if ":" in line:
        split_line = line.split(":")
        speaker_id = split_line[0]
        content = ":".join(split_line[1:]).strip()
    else:
        # Zoom occasionally does not record the speaker ID for a speaker turn.
        # In this case, use <UNK> as placeholder speaker.
        speaker_id = "<UNK>"
        content = line.strip()
    return speaker_id, content

def vtt_to_speakerturns(vtt_file):
    speaker_turns = []
    audio_start_time = 0.0
    with open(vtt_file, 'r') as f:
        next(f)
        speaker_turn = {}
        for i, line in enumerate(f):
            if i % 4 == 0:
                continue

            if i % 4 == 1:
                continue

            if i % 4 == 2:
                audio_start_offset, audio_end_offset = parse_timestamp_vtt(line.rstrip())
                speaker_turn["audio_start_offset"] = audio_start_offset
                speaker_turn["audio_end_offset"] = audio_end_offset
                if i == 2:
                    audio_start_time = audio_start_offset

            if i % 4 == 3:
                speaker_id, content = parse_content_vtt(line.rstrip())
                speaker_turn["speaker_id"] = speaker_id
                speaker_turn["content"] = content
                speaker_turns.append(speaker_turn)
                speaker_turn = {}

    f.close()

    # Normalize start time of first speaker turn to 0.0
    for speaker_turn in speaker_turns:
        speaker_turn["audio_start_offset"] = round(speaker_turn["audio_start_offset"] - audio_start_time, 3)
        speaker_turn["audio_end_offset"] = round(speaker_turn["audio_end_offset"] - audio_start_time, 3)

    return speaker_turns


def parse_speaker_timestamp_rev(line):
    def timestamp_to_secs(timestamp):
        minute, seconds = timestamp.split(":")
        return round(60*float(minute) + float(seconds), 3)

    open_paren_idx = line.index("(")
    close_paren_idx = line.index(")")
    speaker_id = line[:open_paren_idx-1]
    timestamp = line[open_paren_idx+1:close_paren_idx]
    audio_start_offset = timestamp_to_secs(timestamp)

    return speaker_id, audio_start_offset

def rev_to_speakerturns(rev_file):
    speaker_turns = []
    audio_start_time = 0.0
    with open(rev_file, 'r') as f:
        speaker_turn = {}
        for i, line in enumerate(f):
            if i % 3 == 0:
                speaker_id, audio_start_offset = parse_speaker_timestamp_rev(line.rstrip())
                speaker_turn["speaker_id"] = speaker_id
                speaker_turn["audio_start_offset"] = audio_start_offset
            
            if i == 0:
                audio_start_time = audio_start_offset

            if i % 3 == 1:
                speaker_turn["content"] = line.strip()
                speaker_turns.append(speaker_turn)
                speaker_turn = {}
            
            if i % 3 == 2:
                continue
    f.close()

    # Normalize start time of first speaker turn to 0.0
    for speaker_turn in speaker_turns:
        speaker_turn["audio_start_offset"] -= audio_start_time

    # Rev transcripts do not record the end time of the speaker turn.
    # To get around this, hardcode the end of each speaker turn to 0.5 s
    # before the start of the next speaker turn.
    for i in range(len(speaker_turns)-1):
        speaker_turns[i]["audio_end_offset"] = round(speaker_turns[i+1]["audio_start_offset"]-0.5, 3)  
        # speaker_turns[i]["audio_end_offset"] = round(max(speaker_turns[i+1]["audio_start_offset"]-0.5, speaker_turns[i]["audio_start_offset"]+0.6*len(speaker_turns[i]["content"].split())), 3)  
    
    # Address the end time of the last speaker turn.
    # Hardcode the end time to be 0.6 s * number of words spoken.
    speaker_turns[-1]["audio_end_offset"] = round(speaker_turns[-1]["audio_start_offset"]+0.6*len(speaker_turns[-1]["content"].split()), 3)

    return speaker_turns


if __name__ == '__main__':
    # Directories where VTT and Rev txt transcript files are saved
    vtt_dir = "./data/keeper_vtt_clean_part2"
    rev_dir = "./data/keeper_rev_txt_clean_part2"

    json_save_dir = "./data/keeper_json_clean_part2"
    os.makedirs(json_save_dir, exist_ok=True)

    # Process VTT files from Zoom
    vtt_files = sorted(os.listdir(vtt_dir))
    for vtt_file in vtt_files:
        if vtt_file[0] != '.':
            path_to_vtt = os.path.join(vtt_dir, vtt_file)
            speaker_turns = vtt_to_speakerturns(path_to_vtt)
            speaker_turns_ordered = []
            for speaker_turn in speaker_turns:
                ordered = OrderedDict([("audio_start_offset", speaker_turn["audio_start_offset"]),
                                       ("audio_end_offset", speaker_turn["audio_end_offset"]),
                                       ("speaker_id", speaker_turn["speaker_id"]),
                                       ("content", speaker_turn["content"])])
                speaker_turns_ordered.append(ordered)
            
            path_to_json = os.path.join(json_save_dir, vtt_file[:-4]+".json")
            with open(path_to_json, 'w') as outfile:
                json.dump(speaker_turns_ordered, outfile)
            outfile.close()

    # Process txt files from Rev
    rev_files = sorted(os.listdir(rev_dir))
    for rev_file in rev_files:
        if rev_file[0] != '.':
            path_to_rev = os.path.join(rev_dir, rev_file)
            speaker_turns = rev_to_speakerturns(path_to_rev)
            speaker_turns_ordered = []
            for speaker_turn in speaker_turns:
                ordered = OrderedDict([('audio_start_offset', speaker_turn['audio_start_offset']),
                                       ('audio_end_offset', speaker_turn['audio_end_offset']),
                                       ('speaker_id', speaker_turn['speaker_id']),
                                       ('content', speaker_turn['content'])])
                speaker_turns_ordered.append(ordered)

            path_to_json = os.path.join(json_save_dir, rev_file[:-4]+".json")
            with open(path_to_json, 'w') as outfile:
                json.dump(speaker_turns_ordered, outfile)
            outfile.close()


