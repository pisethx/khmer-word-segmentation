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

app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/khmer-word-segmentation")
def get(text: str, method: ISegmentationMethod = ISegmentationMethod.CRF):
    if not text or not method:
        return HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid text or method."
        )

    segmented_text = Segmentation(original_text=text, method=method).segment()

    return HTTPException(
        status_code=HTTP_200_OK,
        detail=(
            {"method": method, "original_text": text, "segmented_text": segmented_text}
        ),
    )
