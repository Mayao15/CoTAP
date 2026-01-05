#!/bin/bash

# 下载COCOStuff27标注文件的脚本
# COCOStuff27是COCO Stuff数据集的27类版本

# 设置变量
DATA_DIR="./data/coco"
ANNOTATIONS_27_DIR="${DATA_DIR}/annotations_27"

echo "开始下载COCOStuff27标注文件..."
echo "注意：COCOStuff27标注需要从COCO Stuff数据集生成"

# 创建目录
mkdir -p "${ANNOTATIONS_27_DIR}/train2017"
mkdir -p "${ANNOTATIONS_27_DIR}/val2017"

# 方法1: 从COCO Stuff数据集生成annotations_27
# 需要下载COCO Stuff标注并转换为27类

# 检查是否已有stuff_annotations_trainval2017.zip
STUFF_ZIP="${DATA_DIR}/stuff_annotations_trainval2017.zip"
if [ -f "${STUFF_ZIP}" ]; then
    echo "找到stuff_annotations_trainval2017.zip，开始解压和转换..."
    
    # 解压到临时目录
    TEMP_DIR=$(mktemp -d)
    unzip -q "${STUFF_ZIP}" -d "${TEMP_DIR}"
    
    # 查找标注文件
    STUFF_ANNOTATIONS_DIR=$(find "${TEMP_DIR}" -type d -name "stuff_annotations" | head -1)
    if [ -z "${STUFF_ANNOTATIONS_DIR}" ]; then
        STUFF_ANNOTATIONS_DIR="${TEMP_DIR}"
    fi
    
    echo "标注文件位置: ${STUFF_ANNOTATIONS_DIR}"
    
    # 检查是否需要转换
    if [ -d "${STUFF_ANNOTATIONS_DIR}/train2017" ] || [ -d "${STUFF_ANNOTATIONS_DIR}/val2017" ]; then
        echo "找到train2017或val2017目录"
        # 可能需要使用Python脚本转换
        echo "提示：可能需要使用Python脚本将COCO Stuff标注转换为27类格式"
    fi
    
    rm -rf "${TEMP_DIR}"
fi

# 方法2: 直接下载预处理的annotations_27（如果可用）
# 注意：这需要从项目仓库或相关资源下载

echo ""
echo "如果上述方法不可用，请尝试以下方法："
echo "1. 从COCO Stuff官方下载标注：http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip"
echo "2. 使用Python脚本将标注转换为27类格式"
echo "3. 或者从项目提供的资源下载预处理好的annotations_27"

# 检查是否有Python转换脚本
if [ -f "scripts/convert_cocostuff_to_27.py" ] || [ -f "convert_cocostuff_to_27.py" ]; then
    echo "找到转换脚本，运行转换..."
    python scripts/convert_cocostuff_to_27.py || python convert_cocostuff_to_27.py
fi

# 验证目录结构
if [ -d "${ANNOTATIONS_27_DIR}/train2017" ] && [ -d "${ANNOTATIONS_27_DIR}/val2017" ]; then
    TRAIN_COUNT=$(find "${ANNOTATIONS_27_DIR}/train2017" -name "*.png" | wc -l)
    VAL_COUNT=$(find "${ANNOTATIONS_27_DIR}/val2017" -name "*.png" | wc -l)
    echo ""
    echo "✓ annotations_27目录已创建"
    echo "  train2017: ${TRAIN_COUNT} 个PNG文件"
    echo "  val2017: ${VAL_COUNT} 个PNG文件"
else
    echo ""
    echo "⚠ 警告: annotations_27目录结构不完整"
    echo "请手动下载并转换COCO Stuff标注文件"
fi


