# Cython 模組，用於加速圖像處理和物件偵測相關運算

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
import cv2

# 必須初始化 numpy C API
np.import_array()

# 定義 C 類型，提高效能 - 使用具體的類型而非通用類型
ctypedef np.int32_t DTYPE_int
ctypedef np.float32_t DTYPE_float
ctypedef np.uint8_t DTYPE_uint8

def resize_image(np.ndarray[DTYPE_uint8, ndim=3] image, int target_size):
    """
    高效能的圖像縮放，保持原始長寬比
    """
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef float scale = min(target_size / <float>w, target_size / <float>h)
    cdef int new_width = int(w * scale)
    cdef int new_height = int(h * scale)
    
    # 縮放圖像
    resized_frame = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # 返回縮放後圖像和比例
    return resized_frame, (w / <float>new_width, h / <float>new_height)

def draw_boxes(np.ndarray[DTYPE_uint8, ndim=3] image, boxes, cls_ids, confidences, names, tuple resize_ratio=(1.0, 1.0)):
    """
    高效繪製邊界框
    
    Args:
        image: 原始圖像 (numpy 數組)
        boxes: 邊界框座標 [x1, y1, x2, y2]
        cls_ids: 類別 ID
        confidences: 信心度
        names: 類別名稱
        resize_ratio: 縮放比例 (x_ratio, y_ratio)
    
    Returns:
        加入邊界框的圖像
    """
    annotated_image = image.copy()
    cdef float x1, y1, x2, y2
    cdef float conf
    cdef int cls_id
    cdef float x_ratio = resize_ratio[0]
    cdef float y_ratio = resize_ratio[1]
    cdef int i
    
    for i in range(len(boxes)):
        # 獲取座標
        x1 = boxes[i][0] * x_ratio
        y1 = boxes[i][1] * y_ratio
        x2 = boxes[i][2] * x_ratio
        y2 = boxes[i][3] * y_ratio
        
        # 獲取類別和信心度
        cls_id = cls_ids[i]
        conf = confidences[i]
        
        # 繪製邊界框
        cv2.rectangle(annotated_image, 
                     (int(x1), int(y1)), 
                     (int(x2), int(y2)), 
                     (0, 255, 0), 2)
        
        # 繪製標籤
        label = f"{names[cls_id]} {conf:.2f}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_width = text_size[0][0]
        text_height = text_size[0][1]
        baseline = text_size[1]
        
        # 確保標籤在圖像範圍內
        label_y = max(int(y1), text_height + 10)
        
        # 繪製背景和文字
        cv2.rectangle(
            annotated_image, 
            (int(x1), label_y - text_height - baseline), 
            (int(x1) + text_width, label_y),
            (0, 255, 0), 
            -1
        )
        cv2.putText(
            annotated_image, 
            label, 
            (int(x1), label_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            2
        )
    
    return annotated_image

def extract_detection_data(result):
    """
    從 YOLO 結果中提取檢測資料
    """
    boxes = []
    cls_ids = []
    confidences = []
    
    try:
        if hasattr(result, 'boxes'):
            boxes_obj = result.boxes
            for i in range(len(boxes_obj)):
                # 獲取並轉換為普通 Python 列表以避免 PyTorch 張量問題
                box = boxes_obj[i].xyxy[0].cpu().numpy().tolist()
                boxes.append(box)
                cls_ids.append(int(boxes_obj[i].cls[0].item()))
                confidences.append(float(boxes_obj[i].conf[0].item()))
    except Exception as e:
        print(f"提取偵測數據時發生錯誤: {e}")
    
    return boxes, cls_ids, confidences
