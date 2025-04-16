import unittest
import random
import os, sys
import configparser

import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler

sys.path.append('../')
sys.path.append('./')

import rdfizer


class TestRdfizer(unittest.TestCase):
    def setUp(self):
        self.nlp = Language()
        self.nlp.add_pipe('entity_ruler')
        self.nlp.from_disk('../language_model')

        self.ruler = self.nlp.pipeline[0][1]

        self.texts = [
                'Example text with no ID',
                'Telequick Mounting plate 300X400',
                'CVS400N TM320D 3P3D',
                '10-24112011-091072',
                'LUS12+LUCB32FU+LUFDA10+LUA1C20',
                'Starter-controller up to 15 kW with Power base LUS12, Control unit LUCMX6BL, Communication module LUFDH11, Accessories LUA1C11'
                ]
        self.ids = [
                [],
                [],
                ['CVS400N', 'TM320D', '3P3D'],
                ['10-24112011-091072'],
                ['LUS12+LUCB32FU+LUFDA10+LUA1C20'],
                ['LUS12', 'LUCMX6BL', 'LUFDH11', 'LUA1C11']
                ]
        # config = configparser.ConfigParser()
        # config.read('../config.ini')
        # for key in config['DEFAULT']:
        #     os.environ[key] = config['DEFAULT'][key]

    def tearDown(self):
        pass

    def test_ner(self):
        for text, ids in zip(self.texts, self.ids):
            nlp = Language()
            doc = nlp(text)
            doc = self.ruler(doc)
            
            self.assertEqual(len(doc.ents), len(ids))
            for ent, id_ in zip(doc.ents, ids):
                if ent.label_ == 'ID':
                    self.assertEqual(ent.text, id_)

    def test_rdfizer(self):
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()

