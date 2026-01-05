#!/bin/bash

# 下载和设置COCO Curated数据集的脚本

# 设置变量
DATA_DIR="./data/coco"
CURATED_DIR="${DATA_DIR}/curated"
DOWNLOAD_URL="https://www.robots.ox.ac.uk/~xuji/datasets/COCOStuff164kCurated.tar.gz"
TAR_FILE="${DATA_DIR}/COCOStuff164kCurated.tar.gz"

echo "开始下载COCO Curated数据集..."
echo "下载地址: ${DOWNLOAD_URL}"
echo "保存位置: ${TAR_FILE}"

# 创建目录（如果不存在）
mkdir -p "${DATA_DIR}"

# 下载文件
if [ ! -f "${TAR_FILE}" ]; then
    echo "正在下载..."
    wget -O "${TAR_FILE}" "${DOWNLOAD_URL}" || {
        echo "下载失败，尝试使用curl..."
        curl -L -o "${TAR_FILE}" "${DOWNLOAD_URL}" || {
            echo "下载失败！请手动下载: ${DOWNLOAD_URL}"
            echo "然后运行: tar -xzf ${TAR_FILE} -C ${DATA_DIR}"
            exit 1
        }
    }
    echo "下载完成！"
else
    echo "文件已存在，跳过下载"
fi

# 解压文件
echo "正在解压..."
if [ -f "${TAR_FILE}" ]; then
    # 先解压到临时目录查看结构
    TEMP_DIR=$(mktemp -d)
    tar -xzf "${TAR_FILE}" -C "${TEMP_DIR}" --strip-components=1 2>/dev/null || tar -xzf "${TAR_FILE}" -C "${TEMP_DIR}"
    
    # 检查解压后的结构
    if [ -d "${TEMP_DIR}/curated" ]; then
        # 如果解压后直接有curated目录，移动它
        mv "${TEMP_DIR}/curated" "${CURATED_DIR}"
    elif [ -d "${TEMP_DIR}/COCOStuff164kCurated/curated" ]; then
        # 如果解压后有COCOStuff164kCurated/curated目录
        mv "${TEMP_DIR}/COCOStuff164kCurated/curated" "${CURATED_DIR}"
    else
        # 尝试查找curated目录
        CURATED_PATH=$(find "${TEMP_DIR}" -type d -name "curated" | head -1)
        if [ -n "${CURATED_PATH}" ]; then
            mv "${CURATED_PATH}" "${CURATED_DIR}"
        else
            echo "警告: 未找到curated目录，尝试直接解压到目标位置"
            tar -xzf "${TAR_FILE}" -C "${DATA_DIR}"
        fi
    fi
    
    # 清理临时目录
    rm -rf "${TEMP_DIR}"
    
    echo "解压完成！"
    
    # 验证目录结构
    if [ -d "${CURATED_DIR}" ]; then
        echo "✓ curated目录已创建: ${CURATED_DIR}"
        echo "目录内容:"
        ls -la "${CURATED_DIR}" | head -10
        
        # 检查必要的文件
        if [ -f "${CURATED_DIR}/val2017/Coco164kFull_Stuff_Coarse_7.txt" ]; then
            echo "✓ 找到必需文件: val2017/Coco164kFull_Stuff_Coarse_7.txt"
        else
            echo "⚠ 未找到文件: val2017/Coco164kFull_Stuff_Coarse_7.txt"
            echo "请检查解压后的目录结构"
        fi
    else
        echo "✗ 错误: curated目录未正确创建"
        exit 1
    fi
else
    echo "错误: 压缩文件不存在"
    exit 1
fi

echo ""
echo "完成！COCO Curated数据集已设置完成"
echo "目录位置: ${CURATED_DIR}"


