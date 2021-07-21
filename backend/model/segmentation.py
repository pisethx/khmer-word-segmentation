from enum import Enum
import re
import util.icu as ICU
import util.symspell as SYMSPELL
import util.crf as CRF
import util.rnn as RNN


class ISegmentationMethod(Enum):
    ICU = "ICU"
    SYMSPELL = "SYMSPELL"
    CRF = "CRF"
    RNN = "RNN"


class Segmentation:
    def __init__(
        self, original_text: str, method: ISegmentationMethod, separator="   "
    ):
        self.original_text = original_text
        self.method = method
        self.separator = separator

    @staticmethod
    def format_result(result: str, separator: str):
        space = " "
        result = (separator).join(
            list(
                filter(
                    lambda x: x != "",
                    result.strip()
                    .replace("   ", space)
                    .replace(" ", space)
                    .replace("\u200b", space)
                    .split(space),
                )
            )
        )

        return result

    def segment(self, separator="   "):
        if not self.original_text or not isinstance(self.original_text, str):
            return []

        text = re.sub("([^\u1780-\u17FF\n ]+)", " \\1 ", self.original_text)

        if self.method == ISegmentationMethod.ICU:
            result = ICU.segment(text)
        elif self.method == ISegmentationMethod.SYMSPELL:
            result = SYMSPELL.segment(text)
        elif self.method == ISegmentationMethod.CRF:
            result = CRF.segment(text)
        elif self.method == ISegmentationMethod.RNN:
            result = RNN.segment(text)

        return Segmentation.format_result(result, separator)
