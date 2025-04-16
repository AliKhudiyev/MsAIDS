import spacy, re, pickle
from spacy.language import Language
from spacy.vocab import Vocab
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler


if __name__ == '__main__':
    nlp = spacy.blank('en')

    ruler = nlp.add_pipe('entity_ruler', config={'validate': True})
    patterns = [{'label': 'ID', 
        'pattern': [{'TEXT': {'REGEX': r'[0-9]+[A-Za-z#\-\$\.:;<>=\?@\\/_\|]+[0-9A-Za-z#\-\$\.:;<>=\?@\\/_\|]{2,}|[A-Za-z#\-\$\.:;<>=\?@\\/_\|]+[0-9]+[0-9A-Za-z#\-\$\.:;<>=\?@\\/_\|]{2,}'}}]}]
    ruler.add_patterns(patterns)
    # id_re = '(?<!\S)(([A-Z]+-*[0-9]+)|([0-9]+-*[A-Z]+))[A-Z0-9]*\\b'

    nlp.to_disk('./language_model')

