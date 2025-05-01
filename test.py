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
        
        # 抑制 ultralytics 的詳細訊息
        import logging
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        
        model = YOLO(model_path)
        print("成功使用 ultralytics API 載入 YOLOv11 模型")
        
        # 嘗試啟用模型優化
        try:
            # 對於 PyTorch 模型，設置為推理模式
            model.model.eval()
            print("模型已設置為推理模式")
        except:
            print("無法設置模型為推理模式，繼續使用默認設定")
            
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

def draw_boxes_on_image(image, results, names, resize_ratio=(1.0, 1.0)):
    """
    在原始圖像上手動繪製邊界框
    
    Args:
        image: 原始圖像
        results: YOLO 檢測結果
        names: 類別名稱
        resize_ratio: 縮放比例 (x_ratio, y_ratio)
    
    Returns:
        加入邊界框的圖像
    """
    annotated_image = image.copy()
    
    # 獲取檢測框
    if hasattr(results[0], 'boxes'):
        boxes = results[0].boxes
        
        # 繪製每個邊界框
        for i in range(len(boxes)):
            # 獲取邊界框座標
            box = boxes[i].xyxy[0].cpu().numpy()  # 確保座標是 numpy 數組
            x1, y1, x2, y2 = box
            
            # 調整比例
            x1 = int(x1 * resize_ratio[0])
            x2 = int(x2 * resize_ratio[0])
            y1 = int(y1 * resize_ratio[1])
            y2 = int(y2 * resize_ratio[1])
            
            # 獲取類別和信心度
            cls_id = int(boxes[i].cls[0].item())
            conf = float(boxes[i].conf[0].item())
            
            # 繪製邊界框
            color = (0, 255, 0)  # 綠色
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # 繪製標籤
            label = f"{names[cls_id]} {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # 確保標籤在圖像範圍內
            label_y = max(y1, text_height + 10)
            
            # 繪製背景和文字
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
    
    # 啟用 OpenCV 優化
    if args.optimize:
        print("啟用 OpenCV 優化...")
        cv2.setUseOptimized(True)
        print(f"OpenCV 硬體加速已啟用: {cv2.useOptimized()}")
    
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
    processed_count = 0
    start_time = time.time()
    fps = 0
    skip_frame_counter = 0
    
    global latest_frame  # 引用全局變量
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("錯誤: 無法從串流獲取畫面。")
                time.sleep(1)  # 等待後重試
                continue
            
            frame_count += 1
            
            # 跳幀處理 - 如果啟用了跳幀且不是處理幀，則跳過偵測
            if args.skip_frames > 0:
                if skip_frame_counter < args.skip_frames:
                    skip_frame_counter += 1
                    with frame_lock:
                        latest_frame = frame.copy()
                    continue
                skip_frame_counter = 0
            
            # 縮放圖像以加速處理，保持原始寬高比
            original_shape = frame.shape
            if args.optimize and args.img_size < frame.shape[1]:
                # 根據長寬比計算新尺寸
                h, w = frame.shape[:2]
                
                # 計算縮放因子，較小的尺寸縮放到目標大小
                scale = min(args.img_size / w, args.img_size / h)
                new_width = int(w * scale)
                new_height = int(h * scale)
                
                # 縮放圖像
                resized_frame = cv2.resize(frame, (new_width, new_height), 
                                          interpolation=cv2.INTER_AREA)
                
                # 儲存縮放比例 (原始/縮放後)
                resize_ratio = (w / new_width, h / new_height)
            else:
                resized_frame = frame
                resize_ratio = (1.0, 1.0)
            
            # 執行物件偵測
            try:
                detection_start = time.time()
                
                if api_type == "ultralytics":
                    # 使用指定的圖像尺寸進行推理，並抑制詳細輸出
                    results = model(resized_frame, conf=args.conf_thres, verbose=False)
                    
                    # 計算偵測時間
                    detection_time = time.time() - detection_start
                    
                    # 根據是否縮放圖像來處理偵測結果
                    if args.optimize and args.img_size < original_shape[1]:
                        # 使用自定義函數在原始圖像上繪製邊界框
                        annotated_frame = draw_boxes_on_image(
                            frame, 
                            results, 
                            results[0].names, 
                            resize_ratio
                        )
                    else:
                        # 直接使用模型輸出的繪製結果
                        annotated_frame = results[0].plot()
                
                # 修正 FPS 顯示
                if processed_count % 5 == 0:
                    end_time = time.time()
                    time_diff = end_time - start_time
                    if time_diff > 0:
                        fps = 5 / time_diff
                    start_time = end_time
                    if args.quiet:  # 只在安靜模式下顯示簡潔的處理速度
                        print(f"處理速度: {fps:.2f} FPS")
                    else:
                        print(f"處理速度: {fps:.2f} FPS, 偵測時間: {detection_time*1000:.0f}ms")
                
                # 在畫面上加入 FPS 和偵測時間
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Speed: {detection_time*1000:.0f}ms", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                processed_count += 1
                
            except Exception as e:
                print(f"偵測過程中發生錯誤: {e}")
                annotated_frame = frame
                # 加入錯誤文字
                cv2.putText(annotated_frame, "ERROR", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 更新全局變量以供Flask使用
            with frame_lock:
                latest_frame = annotated_frame.copy()
                
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
            
            # 短暫延遲以控制 CPU 使用率
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("使用者中斷偵測")
    except Exception as e:
        print(f"發生未預期錯誤: {e}")
    finally:
        cap.release()
        
        # 停止 MJPG-Streamer (如果是我們啟動的)
        if not args.no_streamer and is_process_running('mjpg_streamer'):
            try:
                # 嘗試優雅地結束進程
                mjpg_process = subprocess.run(['pkill', 'mjpg_streamer'], 
                                             stdout=subprocess.PIPE, 
                                             stderr=subprocess.PIPE)
                print("已關閉 MJPG-Streamer")
            except Exception as e:
                print(f"關閉 MJPG-Streamer 時發生錯誤: {e}")
        
        print("串流已關閉")

# 程式結束時的清理函數
def cleanup(*args):
    print("\n正在結束程式...")
    
    # 嘗試停止 MJPG-Streamer
    if is_process_running('mjpg_streamer'):
        try:
            subprocess.run(['pkill', 'mjpg_streamer'], 
                          stdout=subprocess.PIPE, 
                          stderr=subprocess.PIPE)
            print("已關閉 MJPG-Streamer")
        except:
            pass
    
    sys.exit(0)

if __name__ == "__main__":
    # 註冊信號處理器，以便能夠優雅地關閉程式
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    main()
