# Cython 模組，用於加速圖像處理和物件偵測相關運算

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

import numpy as np
cimport numpy as np
import cv2
import requests
import time
from urllib.parse import urlparse

# 必須初始化 numpy C API
np.import_array()

# 定義 C 類型，提高效能
ctypedef np.int32_t DTYPE_int
ctypedef np.float32_t DTYPE_float
ctypedef np.uint8_t DTYPE_uint8

cdef inline float min_float(float a, float b) nogil:
    """內聯函數：返回兩個浮點數中較小的一個"""
    return a if a < b else b

cdef inline int max_int(int a, int b) nogil:
    """內聯函數：返回兩個整數中較大的一個"""
    return a if a > b else b

def resize_image(np.ndarray[DTYPE_uint8, ndim=3] image, int target_size):
    """高效能的圖像縮放，保持原始長寬比"""
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef float scale_w = <float>target_size / <float>w
    cdef float scale_h = <float>target_size / <float>h
    
    # 使用自定義內聯函數計算較小的縮放比例
    cdef float scale = min_float(scale_w, scale_h)
    
    cdef int new_width = <int>(w * scale)
    cdef int new_height = <int>(h * scale)
    
    # 縮放圖像
    cdef np.ndarray resized_frame = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # 返回縮放後圖像和比例
    return resized_frame, (<float>w / <float>new_width, <float>h / <float>new_height)

def draw_boxes(np.ndarray[DTYPE_uint8, ndim=3] image, list boxes, list cls_ids, list confidences, dict names, tuple resize_ratio=(1.0, 1.0)):
    """高效繪製邊界框"""
    cdef np.ndarray[DTYPE_uint8, ndim=3] annotated_image = image.copy()
    cdef int i, x1, y1, x2, y2, cls_id, text_width, text_height, baseline
    cdef int label_y, height_plus_10
    cdef float conf
    cdef float x_ratio = resize_ratio[0]
    cdef float y_ratio = resize_ratio[1]
    cdef int n_boxes = len(boxes)
    cdef tuple color = (0, 255, 0)  # 綠色
    cdef str label
    
    for i in range(n_boxes):
        x1 = <int>(boxes[i][0] * x_ratio)
        y1 = <int>(boxes[i][1] * y_ratio)
        x2 = <int>(boxes[i][2] * x_ratio)
        y2 = <int>(boxes[i][3] * y_ratio)
        
        cls_id = cls_ids[i]
        conf = confidences[i]
        
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        
        label = f"{names[cls_id]} {conf:.2f}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_width = text_size[0][0]
        text_height = text_size[0][1]
        baseline = text_size[1]
        
        height_plus_10 = text_height + 10
        label_y = max_int(y1, height_plus_10)
        
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

def extract_detection_data(object result):
    """從 YOLO 結果中提取檢測資料"""
    cdef list boxes = []
    cdef list cls_ids = []
    cdef list confidences = []
    cdef int i, n_boxes
    
    try:
        if hasattr(result, 'boxes'):
            boxes_obj = result.boxes
            n_boxes = len(boxes_obj)
            
            for i in range(n_boxes):
                box = boxes_obj[i].xyxy[0].cpu().numpy().tolist()
                boxes.append(box)
                cls_ids.append(int(boxes_obj[i].cls[0].item()))
                confidences.append(float(boxes_obj[i].conf[0].item()))
    except Exception as e:
        print(f"提取偵測數據時發生錯誤: {e}")
    
    return boxes, cls_ids, confidences

def calculate_fps(int frame_count, float time_diff):
    """計算 FPS (Frames Per Second)"""
    cdef float fps = 0.0
    if time_diff > 0.001:
        fps = <float>frame_count / time_diff
    return fps

# 移除有問題的 MjpegStreamReader 類和相關函數
# 改為提供更簡單、更穩定的工具函數

def fix_mjpeg_url(str url):
    """修正 MJPEG 流 URL 以提高穩定性"""
    if "?action=stream" in url:
        return url.replace("?action=stream", "?action=snapshot")
    return url

def get_single_jpeg_frame(str url, float timeout=2.0):
    """獲取單幀 JPEG 圖像，避開流解碼問題"""
    cdef np.ndarray frame = None
    cdef bytes content
    
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            content = response.content
            # 使用 OpenCV 解碼 JPEG 圖像
            buffer = np.frombuffer(content, dtype=np.uint8)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"獲取圖像時發生錯誤: {e}")
    
    return frame
