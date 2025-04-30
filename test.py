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
    parser.add_argument('--fps', type=int, default=60,
                        help='相機幀率')
    parser.add_argument('--port', type=int, default=8080,
                        help='MJPG-Streamer 伺服器埠')
    parser.add_argument('--flask-port', type=int, default=5000,
                        help='Flask 網頁服務器埠')
    parser.add_argument('--web-interface', action='store_true', default=True,
                        help='啟用網頁界面')
    return parser.parse_args()

def is_process_running(process_name):
    """檢查指定程序是否正在運行"""
    try:
        # 使用 ps 命令檢查進程
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
        # 使用 Popen 運行命令並讓它在背景執行
        streamer_process = subprocess.Popen(command, 
                                           stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE)
        
        # 給 MJPG-Streamer 一些時間啟動
        time.sleep(2)
        
        # 檢查進程是否仍在運行
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
        # 試著使用 ultralytics API (假設 YOLOv11 使用相同 API)
        from ultralytics import YOLO
        model = YOLO(model_path)
        print("成功使用 ultralytics API 載入 YOLOv11 模型")
        return model, "ultralytics"
    except ImportError:
        try:
            # 如果 ultralytics 不可用，嘗試直接導入 yolov11
            import yolov11
            model = yolov11.load(model_path)
            print("成功使用 yolov11 模組載入模型")
            return model, "native"
        except ImportError:
            raise ImportError("無法導入 YOLOv11。請安裝所需套件。")

# 創建Flask應用
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
        # 獲取最新的處理後畫面
        with frame_lock:
            if latest_frame is not None:
                # 將NumPy數組轉換為JPEG格式
                _, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
                
                # 返回幀作為多部分響應
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 短暫等待，以控制流量
        time.sleep(0.1)

def run_flask_app(port=5000):
    """在單獨的線程中運行Flask應用"""
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

def main():
    args = parse_args()
    
    # 如果沒有指定 --no-streamer 參數，則先啟動 MJPG-Streamer
    if not args.no_streamer:
        print("正在啟動 MJPG-Streamer...")
        if not start_mjpg_streamer(
            device=args.device,
            resolution=args.resolution,
            fps=args.fps, 
            port=args.port
        ):
            print("警告：無法啟動 MJPG-Streamer，將嘗試連接現有串流")
    
    # 建立輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 測試串流連線
    print(f"測試串流連線: {args.stream_url}")
    if not test_stream_connection(args.stream_url):
        print("錯誤: 無法連接到 MJPG 串流。請檢查網址和確認 mjpg-streamer 是否正在運行。")
        return
    
    # 載入 YOLOv11 模型
    try:
        print(f"載入 YOLOv11 模型: {args.model}")
        model, api_type = load_yolo_model(args.model)
    except Exception as e:
        print(f"載入模型時發生錯誤: {e}")
        return
    
    # 如果啟用網頁界面，在背景執行Flask服務器
    if args.web_interface:
        print(f"啟動網頁界面，訪問 http://[您的IP地址]:{args.flask_port} 查看即時偵測結果")
        flask_thread = threading.Thread(target=run_flask_app, args=(args.flask_port,))
        flask_thread.daemon = True  # 設置為守護線程，這樣主程序結束時，此線程也會結束
        flask_thread.start()
    
    # 開啟 MJPG 串流
    print(f"連接串流進行處理: {args.stream_url}")
    cap = cv2.VideoCapture(args.stream_url)
    
    if not cap.isOpened():
        print("錯誤: OpenCV 無法開啟串流。")
        return
    
    print("連接成功。開始物件偵測...")
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    global latest_frame  # 引用全局變量
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("錯誤: 無法從串流獲取畫面。")
                time.sleep(1)  # 等待後重試
                continue
            
            # 執行物件偵測
            try:
                if api_type == "ultralytics":
                    results = model(frame, conf=args.conf_thres)
                    # 在畫面上繪製偵測結果
                    annotated_frame = results[0].plot()
                else:  # api_type == "native"
                    # 請根據 YOLOv11 實際 API 進行調整
                    results = model.detect(frame, conf_threshold=args.conf_thres)
                    annotated_frame = model.draw_detections(frame, results)
                
                # 在畫面上加入 FPS 資訊
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                print(f"偵測過程中發生錯誤: {e}")
                annotated_frame = frame
                # 加入錯誤文字
                cv2.putText(annotated_frame, "偵測錯誤", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 計算 FPS
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                time_diff = end_time - start_time
                fps = 10 / time_diff if time_diff > 0 else 0
                start_time = end_time
                print(f"處理速度: {fps:.2f} FPS")
            
            # 依指定間隔儲存畫面
            if args.save_interval > 0 and frame_count % args.save_interval == 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"detection_{timestamp}.jpg"
                cv2.imwrite(str(output_path), annotated_frame)
                print(f"已儲存偵測結果至 {output_path}")
            
            # 如果啟用，儲存最新畫面
            if args.save_latest:
                latest_path = output_dir / "latest_detection.jpg"
                cv2.imwrite(str(latest_path), annotated_frame)
            
            # 更新全局變量以供Flask使用
            with frame_lock:
                latest_frame = annotated_frame.copy()
                
            # 短暫延遲以控制 CPU 使用率
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("使用者中斷偵測")
    except Exception as e:
        print(f"發生未預期錯誤: {e}")
    finally:
        cap.release()
        print("串流已關閉")

# 程式結束時的清理函數
def cleanup(*args):
    print("\n正在結束程式...")
    # 可以在這裡添加關閉 mjpg-streamer 的代碼，如果需要的話
    sys.exit(0)

if __name__ == "__main__":
    # 註冊信號處理器，以便能夠優雅地關閉程式
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    main()
