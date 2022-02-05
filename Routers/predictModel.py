from fastapi import APIRouter, File, UploadFile

from repository.predict import predictImage

router = APIRouter(
    tags=['Predict Model']
)


@router.post('/predict')
async def predict(file: UploadFile = File(...), model: str = 'potato'):
    return await predictImage(file, model)
