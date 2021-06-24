from enum import Enum
import re
import util.crf as CRF
import util.icu as ICU
import util.symspell as SYMSPELL


class ISegmentationMethod(Enum):
    ICU = "ICU"
    SYMSPELL = "SYMSPELL"
    CRF = "CRF"
    RNN = "RNN"


class Segmentation:
    def __init__(self, original_text: str, method: ISegmentationMethod):
        self.original_text = original_text
        self.method = method

    @staticmethod
    def format_result(result: str):
        space = " "
        seperator = space + space + space
        result = ("   ").join(
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

    def segment(self):
        if not self.original_text or not isinstance(self.original_text, str):
            return []

        text = re.sub("([^\u1780-\u17FF\n ]+)", " \\1 ", self.original_text)

        if self.method == ISegmentationMethod.ICU:
            result = ICU.segment(text)
        elif self.method == ISegmentationMethod.SYMSPELL:
            result = SYMSPELL.segment(text)
        elif self.method == ISegmentationMethod.CRF:
            result = CRF.segment(text)

        return Segmentation.format_result(result)

        # result = (
        #     "ICU : {}\n".format(Segmentation.format_result(ICU.segment(text)))
        #     + "SYMSPELL : {}\n".format(
        #         Segmentation.format_result(SYMSPELL.segment(text))
        #     )
        #     + "CRF : {}\n".format(Segmentation.format_result(CRF.segment(text)))
        # )

        # return result
