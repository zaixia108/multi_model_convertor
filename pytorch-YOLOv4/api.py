import fastapi
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uvicorn
import os
from demo_darknet2onnx import main as darknet2onnx


app = FastAPI()

@app.post("/")
async def create_upload_file(cfg_file: UploadFile = File(...), namesfile: UploadFile = File(...), weight_file: UploadFile = File(...), image_path: UploadFile = File(...), batch_size: int = 1):
    # Save the files
    with open("temp/cfg_file.cfg", "wb") as cfg_file_out:
        cfg_file_out.write(cfg_file.file.read())
    with open("temp/namesfile.names", "wb") as namesfile_out:
        namesfile_out.write(namesfile.file.read())
    with open("temp/weight_file.weights", "wb") as weight_file_out:
        weight_file_out.write(weight_file.file.read())
    with open("temp/image_path.jpg", "wb") as image_path_out:
        image_path_out.write(image_path.file.read())

    batch_size_received = int(batch_size)

    darknet2onnx(cfg_file='temp/cfg_file.cfg',
                 namesfile='temp/namesfile.names',
                 weight_file='temp/weight_file.weights',
                 image_path='temp/image_path.jpg',
                 batch_size=batch_size_received)

    file = None
    files = os.listdir(".")
    for i in files:
        if i.endswith(".onnx"):
            file = i
            break
    return FileResponse(file)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)