import cv2
import numpy as np
import time
import argparse
import os
import requests
import subprocess
import signal
import sys
from pathlib import Path
import threading
from flask import Flask, Response, render_template
import io
import traceback

# 導入 Cython 優化模組
try:
    import yolo_cython_utils
    USE_CYTHON = True
    print("成功載入 Cython 加速模組")
except ImportError:
    USE_CYTHON = False
    print("警告：無法載入 Cython 加速模組，使用純 Python 模式運行")

# 全局變數，用於存儲最新的處理後畫面
latest_frame = None
frame_lock = threading.Lock()

def parse_args():
    parser = argparse.ArgumentParser(description='使用 YOLOv11 在 MJPG 串流上進行物件偵測')
    parser.add_argument('--stream-url', type=str, default='http://localhost:8080/?action=stream', 
                        help='MJPG 串流網址')
    parser.add_argument('--model', type=str, default='yolo11n.pt', 
                        help='YOLOv11 模型路徑')
    parser.add_argument('--conf-thres', type=float, default=0.25, 
                        help='物體偵測信心閾值')
    parser.add_argument('--output-dir', type=str, default='./output', 
                        help='輸出圖片的儲存目錄')
    parser.add_argument('--save-interval', type=int, default=0,
                        help='每 N 個畫面儲存一張偵測結果圖片 (設為 0 停用)')
    parser.add_argument('--save-latest', action='store_true',
                        help='持續更新最新偵測結果圖片')
    parser.add_argument('--no-streamer', action='store_true',
                        help='不自動啟動 MJPG-Streamer')
    parser.add_argument('--device', type=str, default='/dev/video0',
                        help='相機裝置路徑')
    parser.add_argument('--resolution', type=str, default='1280x720',
                        help='相機解析度')
    parser.add_argument('--fps', type=int, default=30,
                        help='相機幀率')
    parser.add_argument('--port', type=int, default=8080,
                        help='MJPG-Streamer 伺服器埠')
    parser.add_argument('--flask-port', type=int, default=5000,
                        help='Flask 網頁服務器埠')
    parser.add_argument('--web-interface', action='store_true', default=True,
                        help='啟用網頁界面')
    parser.add_argument('--img-size', type=int, default=320, 
                        help='模型輸入圖像尺寸，較小尺寸將加快處理速度')
    parser.add_argument('--skip-frames', type=int, default=0, 
                        help='每隔N幀處理一次，0表示處理每一幀')
    parser.add_argument('--optimize', action='store_true', default=True,
                        help='啟用額外優化以提高速度')
    parser.add_argument('--quiet', action='store_true', default=True,
                        help='抑制模型偵測的詳細輸出')
    return parser.parse_args()

def is_process_running(process_name):
    """檢查指定程序是否正在運行"""
    try:
        output = subprocess.check_output(['ps', 'aux']).decode('utf-8')
        return process_name in output
    except subprocess.SubprocessError:
        return False

def start_mjpg_streamer(device='/dev/video0', resolution='1280x720', fps=60, port=8080):
    """啟動 MJPG-Streamer"""
    if is_process_running('mjpg_streamer'):
        print("MJPG-Streamer 已在運行中")
        return True
    
    command = [
        'mjpg_streamer',
        '-i', f'input_uvc.so -d {device} -fps {fps} -r {resolution}',
        '-o', f'output_http.so -w /usr/local/share/mjpg-streamer/www -p {port}'
    ]
    
    try:
        streamer_process = subprocess.Popen(command, 
                                           stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE)
        time.sleep(2)
        if streamer_process.poll() is None:
            print(f"成功啟動 MJPG-Streamer，PID: {streamer_process.pid}")
            return True
        else:
            stdout, stderr = streamer_process.communicate()
            print(f"MJPG-Streamer 啟動失敗: {stderr.decode('utf-8')}")
            return False
    
    except Exception as e:
        print(f"啟動 MJPG-Streamer 時發生錯誤: {e}")
        return False

def test_stream_connection(url, timeout=5):
    """測試 MJPG 串流連線是否可用"""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        if response.status_code == 200:
            print("串流連線測試成功")
            return True
        print(f"串流連線失敗，狀態碼: {response.status_code}")
        return False
    except Exception as e:
        print(f"串流連線測試錯誤: {e}")
        return False

def load_yolo_model(model_path):
    """載入 YOLOv11 模型"""
    try:
        from ultralytics import YOLO
        import logging
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        model = YOLO(model_path)
        print("成功使用 ultralytics API 載入 YOLOv11 模型")
        try:
            model.model.eval()
            print("模型已設置為推理模式")
        except:
            print("無法設置模型為推理模式，繼續使用默認設定")
        return model, "ultralytics"
    except ImportError:
        try:
            import yolov11
            model = yolov11.load(model_path)
            print("成功使用 yolov11 模組載入模型")
            return model, "native"
        except ImportError:
            raise ImportError("無法導入 YOLOv11。請安裝所需套件。")

app = Flask(__name__)

@app.route('/')
def index():
    """主頁路由"""
    return """
    <html>
    <head>
        <title>YOLOv11 即時偵測</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                text-align: center;
                background-color: #f4f4f4;
            }
            h1 {
                color: #333;
            }
            .video-container {
                margin: 20px auto;
                max-width: 1280px;
            }
            img {
                max-width: 100%;
                border: 2px solid #333;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>YOLOv11 即時物件偵測</h1>
        <div class="video-container">
            <img src="/video_feed" />
        </div>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    """提供視頻流給前端頁面"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    """生成視頻流的幀"""
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is not None:
                _, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)

def run_flask_app(port=5000):
    """在單獨的線程中運行Flask應用"""
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

def draw_boxes_on_image(image, results, names, resize_ratio=(1.0, 1.0)):
    """
    在原始圖像上手動繪製邊界框 (Python 版本)
    
    Args:
        image: 原始圖像
        results: YOLO 檢測結果
        names: 類別名稱
        resize_ratio: 縮放比例 (x_ratio, y_ratio)
    
    Returns:
        加入邊界框的圖像
    """
    annotated_image = image.copy()
    
    if hasattr(results[0], 'boxes'):
        boxes = results[0].boxes
        n_boxes = len(boxes)
        
        for i in range(n_boxes):
            box = boxes[i].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(box[0] * resize_ratio[0]), int(box[1] * resize_ratio[1]), int(box[2] * resize_ratio[0]), int(box[3] * resize_ratio[1])
            cls_id = int(boxes[i].cls[0].item())
            conf = float(boxes[i].conf[0].item())
            color = (0, 255, 0)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            label = f"{names[cls_id]} {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_width, text_height = text_size[0]
            baseline = text_size[1]
            label_y = max(y1, text_height + 10)
            cv2.rectangle(
                annotated_image, 
                (x1, label_y - text_height - baseline), 
                (x1 + text_width, label_y),
                color, 
                -1
            )
            cv2.putText(
                annotated_image, 
                label, 
                (x1, label_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                2
            )
    
    return annotated_image

def resize_image_for_detection(image, target_size):
    """圖像縮放函數，適合 Cython 優化"""
    h, w = image.shape[0], image.shape[1]
    scale = min(float(target_size) / float(w), float(target_size) / float(h))
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, (float(w) / float(new_width), float(h) / float(new_height))

def optimize_for_detection(frame, model, args, use_cython=False):
    """優化後的偵測處理流程，更適合 Cython"""
    detection_start = time.time()
    h, w = frame.shape[0], frame.shape[1]
    if args.optimize and args.img_size < w:
        if use_cython:
            resized_frame, resize_ratio = yolo_cython_utils.resize_image(frame, args.img_size)
        else:
            resized_frame, resize_ratio = resize_image_for_detection(frame, args.img_size)
    else:
        resized_frame = frame
        resize_ratio = (1.0, 1.0)
    results = model(resized_frame, conf=args.conf_thres, verbose=False)
    detection_time = time.time() - detection_start
    if args.optimize and args.img_size < w and use_cython:
        boxes, cls_ids, confs = yolo_cython_utils.extract_detection_data(results[0])
        annotated_frame = yolo_cython_utils.draw_boxes(frame, boxes, cls_ids, confs, results[0].names, resize_ratio)
    elif args.optimize and args.img_size < w:
        annotated_frame = draw_boxes_on_image(frame, results, results[0].names, resize_ratio)
    else:
        annotated_frame = results[0].plot()
    return annotated_frame, detection_time

def main():
    args = parse_args()
    use_cython = USE_CYTHON
    
    if not args.no_streamer:
        print("正在啟動 MJPG-Streamer...")
        if not start_mjpg_streamer(
            device=args.device,
            resolution=args.resolution,
            fps=args.fps, 
            port=args.port
        ):
            print("警告：無法啟動 MJPG-Streamer，將嘗試連接現有串流")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"測試串流連線: {args.stream_url}")
    if not test_stream_connection(args.stream_url):
        print("錯誤: 無法連接到 MJPG 串流。請檢查網址和確認 mjpg-streamer 是否正在運行。")
        return
    
    try:
        print(f"載入 YOLOv11 模型: {args.model}")
        model, api_type = load_yolo_model(args.model)
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        return
    
    if args.optimize:
        print("啟用 OpenCV 優化...")
        cv2.setUseOptimized(True)
        print(f"OpenCV 硬體加速已啟用: {cv2.useOptimized()}")
    
    if args.web_interface:
        print(f"啟動網頁界面，訪問 http://[您的IP地址]:{args.flask_port} 查看即時偵測結果")
        flask_thread = threading.Thread(target=run_flask_app, args=(args.flask_port,))
        flask_thread.daemon = True
        flask_thread.start()
    
    print(f"連接串流進行處理: {args.stream_url}")
    cap = cv2.VideoCapture(args.stream_url)
    
    if not cap.isOpened():
        print("錯誤: OpenCV 無法開啟串流。")
        return
    
    print("連接成功。開始物件偵測...")
    
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    fps = 0
    skip_frame_counter = 0
    
    global latest_frame
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("錯誤: 無法從串流獲取畫面。")
                time.sleep(1)
                continue
            
            frame_count += 1
            
            if args.skip_frames > 0:
                if skip_frame_counter < args.skip_frames:
                    skip_frame_counter += 1
                    with frame_lock:
                        latest_frame = frame.copy()
                    continue
                skip_frame_counter = 0
            
            try:
                annotated_frame, detection_time = optimize_for_detection(frame, model, args, use_cython)
                
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Speed: {detection_time*1000:.0f}ms", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                if use_cython:
                    cv2.putText(annotated_frame, "Cython Accelerated", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                
                processed_count += 1
                
                if processed_count % 5 == 0:
                    time_diff = time.time() - start_time
                    if time_diff > 0.001:
                        fps = 5.0 / time_diff
                    start_time = time.time()
                    if not args.quiet:
                        print(f"處理速度: {fps:.2f} FPS, 偵測時間: {detection_time*1000:.0f}ms")
            except Exception as e:
                print(f"偵測過程中發生錯誤: {e}")
                traceback.print_exc()
                annotated_frame = frame
                cv2.putText(annotated_frame, "ERROR", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            with frame_lock:
                latest_frame = annotated_frame.copy()
            
            if args.save_interval > 0 and frame_count % args.save_interval == 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"detection_{timestamp}.jpg"
                cv2.imwrite(str(output_path), annotated_frame)
                print(f"已儲存偵測結果至 {output_path}")
            
            if args.save_latest:
                latest_path = output_dir / "latest_detection.jpg"
                cv2.imwrite(str(latest_path), annotated_frame)
                
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("使用者中斷偵測")
    except Exception as e:
        print(f"發生未預期錯誤: {e}")
    finally:
        cap.release()
        if not args.no_streamer and is_process_running('mjpg_streamer'):
            try:
                subprocess.run(['pkill', 'mjpg_streamer'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
                print("已關閉 MJPG-Streamer")
            except Exception as e:
                print(f"關閉 MJPG-Streamer 時發生錯誤: {e}")
        
        print("串流已關閉")

def cleanup(*args):
    print("\n正在結束程式...")
    
    if is_process_running('mjpg_streamer'):
        try:
            subprocess.run(['pkill', 'mjpg_streamer'], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
            print("已關閉 MJPG-Streamer")
        except:
            pass
    
    os._exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        main()
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"程式發生嚴重錯誤: {e}")
        traceback.print_exc()
        cleanup()
