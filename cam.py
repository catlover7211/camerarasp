import subprocess
import cv2
import time
from ultralytics import YOLO

def start_mjpg_streamer():
    command = [
        'mjpg_streamer',
        '-i', 'input_uvc.so -d /dev/video0 -fps 60 -r 1280x720',
        '-o', 'output_http.so -w /usr/local/share/mjpg-streamer/www -p 8080'
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process

def main():
    # Start mjpg_streamer
    mjpg_process = start_mjpg_streamer()
    
    # Wait for mjpg_streamer to initialize
    # time.sleep(5)
    
    # Read the video stream
    stream_url = 'http://localhost:8080/?action=stream'
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("無法開啟影像流")
        mjpg_process.terminate()
        return
    
    # Load the YOLOv12 model
    try:
        model = YOLO('yolo11n.pt')  # Ensure the path is correct
    except Exception as e:
        print(f"加載模型失敗: {e}")
        cap.release()
        mjpg_process.terminate()
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取影像流")
                break
            
            # Perform object detection with YOLOv12
            try:
                results = model(frame)
                annotated_frame = results[0].plot()  # Process the frame with annotations
                # No cv2.imshow here; frame is processed but not displayed
            except Exception as e:
                print(f"推理失敗: {e}")
                break
            
    except KeyboardInterrupt:
        print("手動終止")
    
    finally:
        # Release resources and terminate mjpg_streamer
        cap.release()
        mjpg_process.terminate()

if __name__ == '__main__':
    main()
