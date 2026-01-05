#!/bin/bash

# 设置COCO标注文件目录结构的脚本
# 将stuff_train2017_pixelmaps和stuff_val2017_pixelmaps中的文件链接到annotations/train2017和annotations/val2017

# 设置变量
DATA_DIR="./data/coco"
ANNOTATIONS_DIR="${DATA_DIR}/annotations"
TRAIN_SOURCE="${ANNOTATIONS_DIR}/stuff_train2017_pixelmaps"
VAL_SOURCE="${ANNOTATIONS_DIR}/stuff_val2017_pixelmaps"
TRAIN_TARGET="${ANNOTATIONS_DIR}/train2017"
VAL_TARGET="${ANNOTATIONS_DIR}/val2017"

echo "开始设置COCO标注文件目录结构..."

# 检查源目录是否存在
if [ ! -d "${TRAIN_SOURCE}" ]; then
    echo "错误: 源目录不存在: ${TRAIN_SOURCE}"
    echo "请先解压stuff_train2017_pixelmaps.zip（如果存在）"
    exit 1
fi

if [ ! -d "${VAL_SOURCE}" ]; then
    echo "错误: 源目录不存在: ${VAL_SOURCE}"
    echo "请先解压stuff_val2017_pixelmaps.zip（如果存在）"
    exit 1
fi

# 创建目标目录
echo "创建目标目录..."
mkdir -p "${TRAIN_TARGET}"
mkdir -p "${VAL_TARGET}"

# 统计源文件数量
TRAIN_COUNT=$(find "${TRAIN_SOURCE}" -name "*.png" | wc -l)
VAL_COUNT=$(find "${VAL_SOURCE}" -name "*.png" | wc -l)

echo "找到 ${TRAIN_COUNT} 个训练集PNG文件"
echo "找到 ${VAL_COUNT} 个验证集PNG文件"

# 创建符号链接（使用符号链接可以节省空间）
echo ""
echo "正在创建符号链接..."

# 获取绝对路径
TRAIN_SOURCE_ABS=$(cd "${TRAIN_SOURCE}" && pwd)
VAL_SOURCE_ABS=$(cd "${VAL_SOURCE}" && pwd)
TRAIN_TARGET_ABS=$(cd "${TRAIN_TARGET}" && pwd)
VAL_TARGET_ABS=$(cd "${VAL_TARGET}" && pwd)

# 为训练集创建符号链接
if [ -d "${TRAIN_TARGET}" ] && [ "$(ls -A ${TRAIN_TARGET} 2>/dev/null)" ]; then
    echo "警告: ${TRAIN_TARGET} 目录已存在且不为空，将跳过已存在的文件"
fi

LINKED_COUNT=0
for file in "${TRAIN_SOURCE_ABS}"/*.png; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        target_file="${TRAIN_TARGET_ABS}/${filename}"
        if [ ! -e "$target_file" ]; then
            ln -s "$file" "$target_file"
            LINKED_COUNT=$((LINKED_COUNT + 1))
            if [ $((LINKED_COUNT % 10000)) -eq 0 ]; then
                echo "训练集: 已处理 ${LINKED_COUNT} 个文件..."
            fi
        fi
    fi
done
echo "训练集: 已创建 ${LINKED_COUNT} 个符号链接"

# 为验证集创建符号链接
if [ -d "${VAL_TARGET}" ] && [ "$(ls -A ${VAL_TARGET} 2>/dev/null)" ]; then
    echo "警告: ${VAL_TARGET} 目录已存在且不为空，将跳过已存在的文件"
fi

LINKED_COUNT=0
for file in "${VAL_SOURCE_ABS}"/*.png; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        target_file="${VAL_TARGET_ABS}/${filename}"
        if [ ! -e "$target_file" ]; then
            ln -s "$file" "$target_file"
            LINKED_COUNT=$((LINKED_COUNT + 1))
        fi
    fi
done
echo "验证集: 已创建 ${LINKED_COUNT} 个符号链接"

# 验证结果
TRAIN_LINKED=$(find "${TRAIN_TARGET}" -name "*.png" | wc -l)
VAL_LINKED=$(find "${VAL_TARGET}" -name "*.png" | wc -l)

echo ""
echo "完成！"
echo "训练集: ${TRAIN_LINKED} 个文件已链接到 ${TRAIN_TARGET}"
echo "验证集: ${VAL_LINKED} 个文件已链接到 ${VAL_TARGET}"

# 检查是否所有文件都已链接
if [ "${TRAIN_LINKED}" -lt "${TRAIN_COUNT}" ]; then
    echo "警告: 训练集文件数量不匹配 (期望: ${TRAIN_COUNT}, 实际: ${TRAIN_LINKED})"
fi

if [ "${VAL_LINKED}" -lt "${VAL_COUNT}" ]; then
    echo "警告: 验证集文件数量不匹配 (期望: ${VAL_COUNT}, 实际: ${VAL_LINKED})"
fi

