#!/bin/bash

# SEKI 實驗環境設置和運行腳本

echo "======================================"
echo "SEKI 實驗環境設置"
echo "======================================"

# 檢查 Python 版本
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✓ Python 已安裝: $python_version"
else
    echo "✗ 錯誤: 未找到 Python3"
    echo "請先安裝 Python 3.7 或更高版本"
    exit 1
fi

# 檢查 pip
if command -v pip3 &> /dev/null; then
    echo "✓ pip3 已安裝"
    pip_cmd="pip3"
elif command -v pip &> /dev/null; then
    echo "✓ pip 已安裝"
    pip_cmd="pip"
else
    echo "✗ 錯誤: 未找到 pip"
    echo "請先安裝 pip"
    exit 1
fi

# 安裝依賴
echo ""
echo "正在安裝依賴項..."
echo "======================================"

if [ -f "requirements.txt" ]; then
    echo "從 requirements.txt 安裝依賴..."
    $pip_cmd install -r requirements.txt
    
    if [[ $? -eq 0 ]]; then
        echo "✓ 依賴項安裝成功"
    else
        echo "✗ 依賴項安裝失敗"
        echo "嘗試手動安裝基本依賴..."
        $pip_cmd install numpy torch scikit-learn matplotlib scipy
    fi
else
    echo "未找到 requirements.txt，手動安裝基本依賴..."
    $pip_cmd install numpy torch scikit-learn matplotlib scipy
fi

echo ""
echo "======================================"
echo "選擇要運行的實驗:"
echo "======================================"
echo "1. 快速演示 (demo.py)"
echo "2. 完整實驗 (run_experiments.py)"
echo "3. 原始玩具實驗 (toy-experiment.py)"
echo "4. 結果可視化 (visualize_results.py)"
echo "5. 退出"
echo ""

while true; do
    read -p "請選擇 (1-5): " choice
    case $choice in
        1)
            echo ""
            echo "運行快速演示..."
            python3 demo.py
            break
            ;;
        2)
            echo ""
            echo "運行完整實驗（這可能需要幾分鐘）..."
            python3 run_experiments.py
            break
            ;;
        3)
            echo ""
            echo "運行原始玩具實驗..."
            python3 toy-experiment.py
            break
            ;;
        4)
            echo ""
            echo "運行結果可視化..."
            python3 visualize_results.py
            break
            ;;
        5)
            echo "退出"
            exit 0
            ;;
        *)
            echo "無效選擇，請選擇 1-5"
            ;;
    esac
done

echo ""
echo "======================================"
echo "實驗完成！"
echo "======================================"
echo ""
echo "生成的文件可能包括："
echo "- *.png: 實驗結果圖表"
echo "- *.json: 實驗數據結果"
echo "- *.md: 實驗報告"
echo ""
echo "查看 README_experiments.md 了解更多信息"
