from icu import BreakIterator, Locale


def gen_khm_words(text: str):
    bi = BreakIterator.createWordInstance(Locale("km"))
    bi.setText(text)
    start = bi.first()
    for end in bi:
        yield text[start:end]
        start = end


def segment(text: str):
    return " ".join(list(gen_khm_words(text)))
