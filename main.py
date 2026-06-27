# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware

# from services.routers.admission_router import router as admission_router
# from services.routers.chat_router import (
#     router as chat_router
# )
# app = FastAPI(
#     title="AI Admission Chatbot",
#     version="5.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# app.include_router(chat_router)
# app.include_router(admission_router)





# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from ultralytics import YOLO
# import numpy as np
# import cv2
# import os
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI(
#     title="Face Detection API",
#     description="Detect faces from uploaded images using YOLO",
#     version="1.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:3000"],  # frontend
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # =========================
# # LOAD MODEL SAFELY
# # =========================

# MODEL_PATH = "yolov8n-face.pt"

# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(
#         "Model file not found. Download yolov8n-face.pt and place it in project folder."
#     )

# model = YOLO(MODEL_PATH)


# # =========================
# # ROUTES
# # =========================

# @app.get("/")
# def root():
#     return {
#         "message": "Face Detection API is running",
#         "docs": "/docs"
#     }


# @app.post("/detect-faces")
# async def detect_faces(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()

#         np_arr = np.frombuffer(contents, np.uint8)
#         image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         if image is None:
#             raise HTTPException(status_code=400, detail="Invalid image file")

#         results = model(image, verbose=False)

#         faces = []

#         for result in results:
#             for box in result.boxes:

#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = float(box.conf[0])

#                 faces.append({
#                     "x": x1,
#                     "y": y1,
#                     "width": x2 - x1,
#                     "height": y2 - y1,
#                     "confidence": round(conf, 4)
#                 })

#         return {
#             "success": True,
#             "faces_detected": len(faces),
#             "faces": faces
#         }

#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "success": False,
#                 "error": str(e)
#             }
#         )






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