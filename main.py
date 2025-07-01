import cv2
import os
import tempfile
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from baseballcv.functions import LoadTools
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()
model = None  # 延遲載入模型用

@app.post("/predict")
async def predict_video(file: UploadFile = File(...)):
    global model
    if model is None:
        try:
            load_tools = LoadTools()
            model_path = load_tools.load_model("ball_tracking")
            model = YOLO(model_path)
        except:
            model_path = 'https://data.balldatalab.com/index.php/s/YkGBwbFtsf34ky3/download/ball_tracking_v4-YOLOv11.pt'
            model = YOLO(model_path)

    # 儲存上傳影片至暫存目錄
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # 執行預測
    box_results = predict_pitch_boxes_from_video_batch(tmp_path, model=model)

    # 刪除暫存影片
    os.remove(tmp_path)

    return JSONResponse(content={"results": box_results})

def predict_pitch_boxes_from_video_batch(video_path, batch_size=16, model=None, device='cpu', aspect_ratio_thresh=(0.75, 1.33)):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    box_results = []

    batch_frames = []
    frame_indices = []

    def filter_box(box):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        width = x2 - x1
        height = y2 - y1
        if height == 0:
            return False
        aspect_ratio = width / height
        return aspect_ratio_thresh[0] <= aspect_ratio <= aspect_ratio_thresh[1]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        batch_frames.append(frame)
        frame_indices.append(frame_idx)
        frame_idx += 1

        if len(batch_frames) == batch_size:
            results = model.predict(source=batch_frames, imgsz=640, device=device, verbose=False)
            for idx, result in enumerate(results):
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    # 找出第一個滿足條件的 box
                    filtered_boxes = [box for box in boxes if filter_box(box)]
                    if filtered_boxes:
                        x1, y1, x2, y2 = filtered_boxes[0].xyxy[0].tolist()
                        box_results.append((frame_indices[idx], [x1, y1, x2, y2]))
                    else:
                        box_results.append((frame_indices[idx], None))
                else:
                    box_results.append((frame_indices[idx], None))
            batch_frames = []
            frame_indices = []

    # 最後一批 frames
    if batch_frames:
        results = model.predict(source=batch_frames, imgsz=640, device=device, verbose=False)
        for idx, result in enumerate(results):
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                filtered_boxes = [box for box in boxes if filter_box(box)]
                if filtered_boxes:
                    x1, y1, x2, y2 = filtered_boxes[0].xyxy[0].tolist()
                    box_results.append((frame_indices[idx], [x1, y1, x2, y2]))
                else:
                    box_results.append((frame_indices[idx], None))
            else:
                box_results.append((frame_indices[idx], None))

    cap.release()
    return box_results

# 若本地測試時使用
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
