"""
提供更穩健的 MJPEG 串流處理工具
"""

import cv2
import numpy as np
import requests
import time
from urllib.parse import urlparse
import threading
import queue

class RobustMJPEGReader:
    """更穩健的 MJPEG 串流讀取器，處理各種解碼問題"""
    
    def __init__(self, stream_url, buffer_size=10):
        """
        初始化 MJPEG 讀取器
        
        Args:
            stream_url: MJPEG 串流 URL
            buffer_size: 緩衝區大小（幀數）
        """
        self.stream_url = stream_url
        self.running = False
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.current_frame = None
        self.last_good_frame = None
        self.frame_count = 0
        self.error_count = 0
        self.reader_thread = None
        
        # 嘗試獲取單一幀來確認串流是否正常
        self._test_connection()
        
    def _test_connection(self):
        """測試串流連線"""
        try:
            if "?action=stream" in self.stream_url:
                # 對於 mjpg-streamer，使用 snapshot 檢查連線
                test_url = self.stream_url.replace("?action=stream", "?action=snapshot")
                response = requests.get(test_url, timeout=5)
                if response.status_code == 200:
                    # 確認是否為有效圖像
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None and img.size > 0:
                        print("MJPEG 串流連線測試成功")
                        return True
            
            # 回退方式：嘗試使用 OpenCV 直接開啟
            cap = cv2.VideoCapture(self.stream_url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    print("MJPEG 串流連線測試成功")
                    return True
            
            print("警告: MJPEG 串流連線測試失敗")
            return False
        except Exception as e:
            print(f"測試 MJPEG 串流連線時發生錯誤: {e}")
            return False
    
    def _reader_worker(self):
        """背景讀取串流的工作線程"""
        cap = None
        reconnect_delay = 1.0  # 初始重連延遲（秒）
        max_reconnect_delay = 30.0  # 最大重連延遲
        
        while self.running:
            try:
                # 如果還未開啟或需要重新連線
                if cap is None or not cap.isOpened():
                    print(f"正在連接 MJPEG 串流: {self.stream_url}")
                    cap = cv2.VideoCapture(self.stream_url)
                    
                    # 設定 OpenCV 參數以處理 APP 欄位解碼錯誤
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                    
                    # 如果無法開啟，稍後重試
                    if not cap.isOpened():
                        print(f"無法開啟 MJPEG 串流，{reconnect_delay} 秒後重試")
                        time.sleep(reconnect_delay)
                        # 指數退避策略
                        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                        continue
                    else:
                        # 連線成功，重置重連延遲
                        reconnect_delay = 1.0
                
                # 讀取一幀
                ret, frame = cap.read()
                
                if ret and frame is not None and frame.size > 0:
                    # 成功讀取幀
                    self.frame_count += 1
                    self.last_good_frame = frame.copy()
                    
                    # 將幀放入緩衝區，如果緩衝區已滿則丟棄舊幀
                    if self.frame_buffer.full():
                        try:
                            self.frame_buffer.get_nowait()
                        except queue.Empty:
                            pass
                    
                    self.frame_buffer.put(frame)
                else:
                    # 讀取失敗，記錄錯誤
                    self.error_count += 1
                    print(f"讀取 MJPEG 串流幀失敗 ({self.error_count})")
                    
                    # 如果連續失敗次數過多，重新連接
                    if self.error_count >= 5:
                        print("連續多次讀取失敗，嘗試重新連接...")
                        if cap is not None:
                            cap.release()
                        cap = None
                        self.error_count = 0
                
                # 短暫休眠以避免 CPU 使用率過高
                time.sleep(0.01)
                
            except Exception as e:
                # 處理任何異常
                print(f"MJPEG 讀取線程發生錯誤: {e}")
                self.error_count += 1
                
                # 關閉當前連接並嘗試重新連接
                if cap is not None:
                    cap.release()
                cap = None
                time.sleep(reconnect_delay)
        
        # 停止時釋放資源
        if cap is not None and cap.isOpened():
            cap.release()
    
    def start(self):
        """開始讀取串流"""
        if not self.running:
            self.running = True
            self.reader_thread = threading.Thread(target=self._reader_worker)
            self.reader_thread.daemon = True
            self.reader_thread.start()
            print("已啟動 MJPEG 串流讀取線程")
    
    def stop(self):
        """停止讀取串流"""
        self.running = False
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=1.0)
        print("已停止 MJPEG 串流讀取線程")
    
    def read(self):
        """讀取目前的幀"""
        if not self.running:
            return False, None
        
        try:
            # 嘗試從緩衝區獲取最新幀
            frame = self.frame_buffer.get_nowait()
            self.current_frame = frame
            return True, frame
        except queue.Empty:
            # 如果緩衝區為空但有最後的好幀，則返回最後的好幀
            if self.last_good_frame is not None:
                return True, self.last_good_frame.copy()
            # 否則返回失敗
            return False, None
