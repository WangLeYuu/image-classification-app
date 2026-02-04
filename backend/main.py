from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from model import get_classifier
import uvicorn


app = FastAPI(
    title="Image Classification API",
    description="使用 MobileNetV2 进行图像分类的 RESTful API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    def get_root_info():
        return {
            "message": "欢迎使用图像分类 API",
            "endpoints": {
                "health": "/health",
                "predict": "/predict"
            }
        }
    return get_root_info()


@app.get("/health")
def health_check():
    def get_health_status():
        return {
            "status": "healthy",
            "service": "image-classification-api"
        }
    return get_health_status()


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    def validate_file_type(filename):
        allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        return any(filename.lower().endswith(ext) for ext in allowed_extensions)

    if not validate_file_type(file.filename):
        raise HTTPException(
            status_code=400,
            detail="不支持的文件类型。请上传图片文件（jpg, jpeg, png, bmp, gif, webp）"
        )

    try:
        contents = await file.read()
        classifier = get_classifier()
        result = classifier.predict(contents)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"图像分类失败: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
