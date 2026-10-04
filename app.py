from fastapi import FastAPI
from fastapi.responses import FileResponse

app = FastAPI(
    title="3D Microscopy Segmentation"
)


@app.get("/")
def home():

    return FileResponse("index.html")


@app.get("/health")
def health():

    return {
        "status": "3D Microscopy Segmentation API is running"
    }