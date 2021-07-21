from fastapi import FastAPI
from pydantic import BaseModel

from starlette.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.status import (
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_200_OK,
)

from model.segmentation import Segmentation, ISegmentationMethod
from fastapi.middleware.cors import CORSMiddleware

import os
import glob
import re
from util.rnn import TRAIN_FILE, cleanup_str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/khmer-word-segmentation")
def get(text: str, method: ISegmentationMethod = ISegmentationMethod.RNN):
    if not text or not method:
        return HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid text or method."
        )

    segmented_text = Segmentation(original_text=text, method=method).segment()

    return HTTPException(
        status_code=HTTP_200_OK,
        detail=(
            {
                "method": method,
                "segmented_text": segmented_text,
                # "original_text": text,
            }
        ),
    )


@app.get("/test")
def get():
    file = open(TRAIN_FILE, "r", encoding="utf-8")
    lines = []
    segmented_lines = []

    for x in file:
        lines.append(cleanup_str(x.strip().replace(" ", "")))
        segmented_lines.append(cleanup_str(x.strip(), "."))

    # data_dir = "kh_data_" + '100'
    # system_dir = os.path.join("util", data_dir)
    # path = system_dir + "/*_seg_200b.txt"
    # files = glob.glob(path)

    # for file in files:
    #     filenum = re.search(r"\d+_", file).group(0)
    #     f = open(file, "r")
    #     rl = f.readlines()
    #     f.close()
    #     lines.append(''.join(rl).strip("\u200b").strip())

    #     f = open(file.replace("_seg_200b.txt", "_orig.txt"), "r")
    #     rl = f.readlines()
    #     f.close()
    #     segmented_lines.append(''.join(rl).strip("\u200b").strip())

    corrected = {
        "icu": 0,
        "sym": 0,
        "crf": 0,
        "rnn": 0,
    }

    for idx in range(len(lines)):
        text = lines[idx]
        separator = "."
        icu = Segmentation(original_text=text, method=ISegmentationMethod.ICU).segment(
            separator=separator
        )
        sym = Segmentation(
            original_text=text, method=ISegmentationMethod.SYMSPELL
        ).segment(separator=separator)
        crf = Segmentation(original_text=text, method=ISegmentationMethod.CRF).segment(
            separator=separator
        )
        rnn = Segmentation(original_text=text, method=ISegmentationMethod.RNN).segment(
            separator=separator
        )

        segmented_str = Segmentation.format_result(
            result=segmented_lines[idx], separator=separator
        )

        if icu == segmented_str:
            corrected["icu"] += 1
        if sym == segmented_str:
            corrected["sym"] += 1
        if rnn == segmented_str:
            corrected["rnn"] += 1
        if crf == segmented_str:
            corrected["crf"] += 1

    return HTTPException(
        status_code=HTTP_200_OK,
        detail=(
            {
                "total": len(lines),
                "corrected": corrected,
            }
        ),
    )
