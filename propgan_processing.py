import os
import re
import pdb

class SentenceElement(object):
    def __init__(self, text, label, article_id, sentence_id):
        self.text = text
        self.label = label
        self.article_id = article_id
        self.sentence_id = sentence_id

label_mapping = {"non-propaganda": 0, "propaganda": 1}

def read_files(data_path, id):
    sentences = []

    article_path = "{}/train-articles/article{}.txt".format(data_path, id)
    label_path = "{}/train-labels-SLC/article{}.task-SLC.labels".format(data_path, id)

    with open(article_path, 'r') as file:
        article_lines = file.readlines()

    with open(label_path, 'r') as file:
        label_lines = file.readlines()

    assert len(article_lines) == len(label_lines)

    for x, y in zip(article_lines, label_lines):

        if x == "\n": # skip those empty lines
            continue

        items = y.split()
        article_id = int(items[0])
        sentence_id = int(items[1])
        label = label_mapping[items[2]]

        text = x.strip()

        sentence = SentenceElement(text, label, article_id, sentence_id)

        sentences.append(sentence)

    return sentences

def load_sentences(data_path):
    file_names = os.listdir("{}/train-articles/".format(data_path))
    sentences = []
    for name in file_names:
        _id = re.sub("[^0-9]", "", name)
        sentence = read_files(data_path=data_path, id=_id)
        sentences.extend(sentence)

    return sentences

def main():
    pdb.set_trace()
    sentences = load_sentences("datasets")

if __name__ == "__main__":
    main()
