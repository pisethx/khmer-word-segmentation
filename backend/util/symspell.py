from symspellpy.symspellpy import SymSpell
import os

KM_DICT = "util/km_KH.dic"

# Symspell
def create_dictionary(self, corpus):
    with open(corpus) as f:
        print("Creating Dictionary...")
        for word in f:
            self.create_dictionary_entry(word, 1)
        self.save_pickle(corpus + ".pickle")
        print("Done.\n")


def segment(text):
    sym = SymSpell()

    if not os.path.exists(KM_DICT + ".pickle"):
        create_dictionary(sym, KM_DICT)

    sym.load_pickle(KM_DICT + ".pickle")
    result = sym.word_segmentation(text)
    return result.segmented_string
