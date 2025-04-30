import cv2
from flask import Flask, Response
from ultralytics import YOLO

app = Flask(__name__)

def generate_frames():
    # 假設視頻源來自 mjpg_streamer，若使用攝像頭可改為 cv2.VideoCapture(0)
    stream_url = 'http://localhost:8080/?action=stream'
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("無法開啟影像流")
        return
    
    # 加載 YOLO 模型
    model = YOLO('yolo11n.pt')  # 確保模型文件路徑正確
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取影像流")
            break
        
        # 使用 YOLO 進行影像辨識
        results = model(frame)
        annotated_frame = results[0].plot()  # 獲取帶有標註的圖像
        
        # 將幀轉換為 JPEG 格式
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        
        # 構建 MJPEG 流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 啟動 Flask 服務器，運行在端口 5001（避免與其他服務衝突）
    app.run(host='0.0.0.0', port=5001)
