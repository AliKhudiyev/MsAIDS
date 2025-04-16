import spacy, re
from spacy.language import Language
from spacy.vocab import Vocab
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
from spacy.tokens import Doc
import pickle


text = '''Here is an example Hello, Spacy!'''
nlp = spacy.blank('en') # Language(Vocab())
# nlp = spacy.blank('en')


def create_tokenizer(nlp):
    special_cases = {}
    # prefix_re = re.compile(r'H') # re.compile(r'^H?://[a-z]+')
    # suffix_re = re.compile(r' ')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)
    infix_re = re.compile(r'H[a-z]+')
    token_re = re.compile(r'H[a-z]+')
    url_re = re.compile(r'')

    return Tokenizer(nlp.vocab, 
            rules=special_cases, 
            prefix_search=None, # prefix_re.search,
            suffix_search=None, # suffix_re.search,
            infix_finditer=None, # infix_re.finditer,
            token_match=token_re.match,
            url_match=url_re.match)
            

class MyTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        ids = re.findall(r'H[a-z]+', text)
        
        doc = Doc(self.vocab, words=words)
        doc.user_data['ids'] = ids

        return doc
    
    def to_bytes(self, exclude):
        return pickle.dumps(self.__dict__)

    def from_bytes(self, data, exclude):
        return self.__dict__.update(pickle.loads(data))

    def to_disk(self, path, exclude):
        with open(path, 'wb') as f:
            f.write(self.to_bytes(exclude))

    def from_disk(self, path, exclude):
        with open(path, 'rb') as f:
            self.from_bytes(f.read(), exclude)


# tokenizer = create_tokenizer(nlp)
# nlp.tokenizer = tokenizer
# nlp.tokenizer = Tokenizer(nlp.vocab) # MyTokenizer(nlp.vocab)
mytkr = MyTokenizer(nlp.vocab)
nlp.tokenizer = mytkr # MyTokenizer(nlp.vocab)
nlp.to_disk('lang_model2')

doc_tmp = nlp(text)
print([t for t in doc_tmp])

# data = nlp.to_bytes()
# with open('mymodel', 'wb') as f:
#     f.write(data)
# 
# data2 = None
# with open('mymodel', 'rb') as f:
#     data2 = f.read()
# 
# nlp.from_bytes(data2)

# nlp = spacy.from_disk('lang_model2')
nlp.from_disk('lang_model2')
print(nlp.pipeline)
doc = nlp(text)

print([t.text for t in doc])
print(doc.user_data['ids'])

