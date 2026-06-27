



# live capturing 


from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n-face.pt")

# =========================
# LAPTOP CAMERA (OPTIONAL MODE)
# =========================
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def laptop_stream():
    while True:
        ret, frame = camera.read()
        if not ret:
            continue

        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.get("/laptop-feed")
def laptop_feed():
    return StreamingResponse(laptop_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# =========================
# MOBILE CAMERA STREAM (WEBSOCKET)
# =========================
@app.websocket("/mobile-stream")
async def mobile_stream(websocket: WebSocket):
    await websocket.accept()

    while True:
        try:
            data = await websocket.receive_bytes()

            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            results = model(frame, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{conf:.2f}",
                                (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,255,0), 2)

            _, buffer = cv2.imencode(".jpg", frame)

            await websocket.send_bytes(buffer.tobytes())

        except Exception:
            break