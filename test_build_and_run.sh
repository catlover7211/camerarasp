#!/bin/bash

echo "編譯 Cython 模組..."
python3 setup.py build_ext --inplace

# 設置 OpenCV 環境變量以增強 JPEG 解碼容錯性
export OPENCV_MJPEG_DECODE_TIMEOUT=0
export OPENCV_JPEG_IGNORE_LIMITS=1
# 設置 OpenCV 不顯示錯誤訊息
export OPENCV_LOG_LEVEL=0
export OPENCV_QUIET=1

# 處理 SIGINT 信號，避免腳本被中斷時仍有 Python 進程運行
trap 'echo "收到中斷信號，正在終止程式..."; pkill -f "python3 test.py"; exit 0' INT TERM

echo "運行物件偵測程式..."
# 將標準錯誤重定向到 /dev/null 以抑制錯誤訊息
python3 test.py --skip-frames 2 "$@" 2>/dev/null

# 確保程式正常退出
exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "程式以錯誤代碼 $exit_code 退出"
    # 確保沒有剩餘的 Python 進程
    pkill -f "python3 test.py" 2>/dev/null
fi

exit $exit_code
