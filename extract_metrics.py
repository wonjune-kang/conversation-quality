import sys
import os
import json
import csv
from tqdm import tqdm

from utils.lexical_metrics import get_base_metrics
from utils.sentiment import get_vader_sentiment_scores
from utils.responsivity import init_spacy, init_sentence_encoder, responsivity_metrics


METRIC_NAMES = ["Conversation ID", "Words per hour", "Speech per turn",
                "Mean inter-speaker silence", "Turn taking balance",
                "Interruption rate", "Grade level",
                "MATTR lexical diversity", "Mean word length",
                "VADER sentiment", "Responsivity rate"]

nlp = init_spacy()
encoder = init_sentence_encoder()


def get_metric_values_list(snippets):
    base_metrics = get_base_metrics(snippets)
    base_metric_vals = [x[1] for x in base_metrics]

    vader_sentiment = get_vader_sentiment_scores(snippets)
    avg_vader_sentiment = sum([x[1] for x in vader_sentiment])/len(vader_sentiment)

    _, _, response_speaker_turns = responsivity_metrics(snippets, nlp, encoder)
    responsivity_rate = len(response_speaker_turns)/len(snippets)

    return base_metric_vals + [round(avg_vader_sentiment, 3)] + [round(responsivity_rate,3)]

if __name__ == '__main__':
    json_dir = sys.argv[1]
    csv_save_file = sys.argv[2]

    with open(csv_save_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t')
        csv_writer.writerow(METRIC_NAMES)

        for json_file in tqdm(sorted(os.listdir(json_dir), key=lambda x: int(x[:-5]))):
            with open(os.path.join(json_dir, json_file), 'r') as f:
                snippets = json.load(f)
            f.close()

            metric_vals = get_metric_values_list(snippets)
            csv_writer.writerow([json_file[:-5]]+metric_vals)
