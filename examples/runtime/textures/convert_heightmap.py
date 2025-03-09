import cv2
import numpy as np
import argparse
import os

def convert_nonstandard_heightmap(input_path, output_path, min_height=-100, max_height=45000, output_mode='rgb'):
    """
    将非标准高度图转换为标准PNG格式高度图

    参数:
        input_path: 输入的非标准高度图路径
        output_path: 输出的标准高度图路径
        min_height: 最小高度值
        max_height: 最大高度值
        output_mode: 输出模式 ('gray8', 'gray16', 'rgb')
    """
    # 读取输入图像
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"无法读取图像: {input_path}")

    # 解码高度值
    if img.dtype == np.uint8 and len(img.shape) == 3 and img.shape[2] == 4:  # CV_8UC4
        # 直接转换为32位浮点型
        heightmap = img.view(np.float32).reshape(img.shape[0], img.shape[1])
        print("处理8位4通道图像 (CV_8UC4)")
    else:
        # 创建空白的32位浮点高度图
        heightmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        # 解码高度值
        lower_count = 0
        upper_count = 0
        abnormal_count = 0

        # 遍历每个像素
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                b, g, r = img[row, col] if len(img.shape) == 3 else (img[row, col], 0, 0)

                # 与C++代码中相同的解码算法
                new_val_int = (r << 16) | (g << 8) | b

                # 检查第23位，判断正负
                if (new_val_int & (0x01 << 23)):
                    new_val_int &= ~(0x01 << 23)
                    new_val_int = -new_val_int

                # 处理特殊值
                if new_val_int in [-1000000, 1000000, -500000]:
                    if new_val_int == -500000:
                        new_val = -50000
                    else:
                        new_val = new_val_int
                else:
                    # 普通值除以100.0
                    new_val = float(new_val_int / 100.0)

                # 应用范围限制
                if new_val <= min_height:
                    lower_count += 1
                    new_val = min_height
                elif np.isnan(new_val) or np.isinf(new_val):
                    abnormal_count += 1
                    new_val = min_height
                elif new_val >= max_height:
                    upper_count += 1
                    new_val = max_height

                heightmap[row, col] = new_val

        print(f"处理结果统计: {lower_count}个低值点, {upper_count}个高值点, {abnormal_count}个异常值")

    # 高斯模糊处理，与原代码保持一致
    heightmap = cv2.GaussianBlur(heightmap, (5, 5), 0)

    # 计算高度范围
    valid_mask = ~(np.isnan(heightmap) | np.isinf(heightmap))
    if np.any(valid_mask):
        actual_min = np.min(heightmap[valid_mask])
        actual_max = np.max(heightmap[valid_mask])
        print(f"高度范围: {actual_min} 到 {actual_max}")

    # 根据输出模式生成PNG格式图像
    if output_mode == 'gray8':
        # 8位灰度图 (0-255)
        normalized = np.zeros_like(heightmap)
        cv2.normalize(heightmap, normalized, 0, 255, cv2.NORM_MINMAX)
        output_img = normalized.astype(np.uint8)
        print("输出8位灰度PNG (精度降低)")

    elif output_mode == 'gray16':
        # 16位灰度图 (0-65535)
        normalized = np.zeros_like(heightmap)
        cv2.normalize(heightmap, normalized, 0, 65535, cv2.NORM_MINMAX)
        output_img = normalized.astype(np.uint16)
        print("输出16位灰度PNG (中等精度)")

    elif output_mode == 'rgb':
        # 将高度值编码到RGB通道
        height_range = max_height - min_height
        normalized = (heightmap - min_height) / height_range if height_range > 0 else heightmap - min_height

        # 限制到0-1范围
        # normalized = np.clip(normalized, 0, 1)

        # 将高度值编码为24位RGB
        # 缩放到0-16777215范围 (2^24 - 1)
        values = (normalized * 16777215).astype(np.uint32)

        # 分离通道
        r = (values >> 16) & 0xFF
        g = (values >> 8) & 0xFF
        b = values & 0xFF

        # 创建RGB图像
        output_img = np.zeros((heightmap.shape[0], heightmap.shape[1], 3), dtype=np.uint8)
        output_img[..., 0] = b
        output_img[..., 1] = g
        output_img[..., 2] = r
        print("输出RGB编码PNG (高精度)")

    else:
        raise ValueError(f"不支持的输出模式: {output_mode}")

    # 保存结果
    cv2.imwrite(output_path, output_img)

    # 保存可视化图像
    vis_path = os.path.splitext(output_path)[0] + "_visualization.png"
    vis_img = np.zeros_like(heightmap)
    cv2.normalize(heightmap, vis_img, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(vis_path, vis_img.astype(np.uint8))
    print(f"保存可视化图像到: {vis_path}")

    return output_img

def main():
    parser = argparse.ArgumentParser(description='将非标准高度图转换为标准PNG格式高度图')
    parser.add_argument('input', help='输入的非标准高度图路径')
    parser.add_argument('output', help='输出的标准PNG高度图路径')
    parser.add_argument('--min', type=float, default=-100, help='最小高度值')
    parser.add_argument('--max', type=float, default=45000, help='最大高度值')
    parser.add_argument('--mode', choices=['gray8', 'gray16', 'rgb'], default='rgb',
                        help='输出模式: gray8(8位灰度), gray16(16位灰度), rgb(RGB编码)')

    args = parser.parse_args()

    try:
        convert_nonstandard_heightmap(args.input, args.output, args.min, args.max, args.mode)
        print(f"成功将非标准高度图转换为标准PNG格式高度图: {args.output}")

        # 打印解码说明
        if args.mode == 'rgb':
            print("\n解码说明:")
            print("这个RGB编码的高度图可通过以下方式解码:")
            print("height = min_height + (r*65536 + g*256 + b) * (max_height - min_height) / 16777215")
            print(f"其中 min_height={args.min}, max_height={args.max}")

    except Exception as e:
        print(f"转换失败: {str(e)}")

if __name__ == "__main__":
    main()
