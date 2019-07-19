import tokenization
import pdb

"""
The (Official) BERT Tokenizer does the following:

1) Text normalization: Convert all whitespace characters to spaces,
and (for the Uncased model) lowercase the input and strip out accent markers.
E.g., John Johanson's, → john johanson's,.

2) Punctuation splitting: Split all punctuation characters on both sides
(i.e., add whitespace around all punctuation characters). Punctuation characters are defined
as (a) Anything with a P* Unicode class, (b) any non-letter/number/space ASCII character
(e.g., characters like $ which are technically not punctuation).
E.g., john johanson's, → john johanson ' s ,

3) WordPiece tokenization: Apply whitespace tokenization to the output of the above procedure,
and apply WordPiece tokenization to each token separately. (Our implementation is directly
based on the one from tensor2tensor, which is linked).
E.g., john johanson ' s , → john johan ##son ' s ,
"""

class BertTokenizer(object):
    def __init__(self, vocab_file, do_lower_case=True):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    def raw2tokens(self, text):
        return self.tokenizer.tokenize(text)

    def tokens2ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


def read_movie_reviews(path):
    with open(path) as file:
        lines = file.readlines()
        header = lines[0]
        body = lines[1:]
    phrases = []
    sentiments = []
    for line in body:
        line = line.strip().split('\t')
        phrase = line[2]
        sentiment = line[3]
        phrases.append(phrase)
        sentiments.append(sentiment)

    return phrases, sentiments

def test():
    path = "/home/punpun/Desktop/sentiment1_data/train.tsv"
    phrases, sentiments = read_movie_reviews(path)


if __name__ == "__main__":
    test()
