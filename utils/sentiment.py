import sys
import os
import json
from collections import defaultdict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def get_vader_sentiment_scores(snippets):
    '''
    Returns list of tuples (speaker_id, vader_compound_score) with length equal
    to the total number of speaker turns (length of snippets).

    VADER sentiment scoring:
        positive sentiment: compound score >= 0.05
        neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
        negative sentiment: compound score <= -0.05
    '''
    analyzer = SentimentIntensityAnalyzer()
    speaker_sentiments = []

    for speaker_turn in snippets:
        content = speaker_turn['content']
        if "[" in content:
            content = content[:content.index("[")] + content[content.index("]")+1:]
        sentiment = analyzer.polarity_scores(content)
        speaker_sentiments.append((speaker_turn["speaker_id"], sentiment["compound"]))

    return speaker_sentiments

def parse_emolex(path_to_emolex='./data/emotion_lexicons'):
    """
    Returns:
        word2sentiments: dict
            key: str
            value: set of strs
        word2emotions: dict
            key: str
            value: set of strs
    """
    FP = os.path.join(path_to_emolex, 'NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')

    word2sentiments = defaultdict(set)
    word2emotions = defaultdict(set)
    f = open(FP, 'r')
    for i, line in enumerate(f.readlines()):
        if i > 1:
            # line: 'abandoned\tanger\t1\n',
            word, emotion, flag = line.strip('\n').split('\t')
            if int(flag) == 1:
                if emotion == 'positive' or emotion == 'negative':
                    word2sentiments[word].add(emotion)
                else:
                    word2emotions[word].add(emotion)
    f.close()

    return word2sentiments, word2emotions

def get_emolex_scores(snippets, word2sentiments, word2emotions):
    '''
    Returns list of tuples (speaker_id, emotion_list) with length equal
    to the total number of speaker turns (length of snippets).

    emotion_list is a list of emotions that are reflected in the words of
    that speaker turn, as determined by Emolex; it can be empty.
    '''
    speaker_turn_emotions = []

    for speaker_turn in snippets:
        content = speaker_turn['content']
        if "[" in content:
            content = content[:content.index("[")] + content[content.index("]")+1:]
        emotions = set()
        for word in content.split():
            if word in word2emotions:
                emotions.update(word2emotions[word])

        speaker_turn_emotions.append((speaker_turn['speaker_id'], sorted(list(emotions))))

    return speaker_turn_emotions


if __name__ == "__main__":
    json_file = sys.argv[1]
    with open(json_file, 'r') as f:
        snippets = json.load(f)
    f.close()

    word2sentiments, word2emotions = parse_emolex()
    speaker_turn_emolex = get_emolex_scores(snippets, word2sentiments, word2emotions)
    speaker_turn_vader = get_vader_sentiment_scores(snippets)

    print(speaker_turn_emolex)
    print(speaker_turn_vader)




