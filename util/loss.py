import torch
import torch.nn.functional as F

def differentiable_colorfulness(images):
    """
    计算单批次图像的 Colorfulness (PyTorch 可微版)
    images: 形状为 [B, 3, H, W] 的 RGB 张量，取值范围应为 [0, 1]
    """
    # 放大到 0-255，以适配 Hasler & Suesstrunk 公式的经验常数
    img = images * 255.0
    
    R = img[:, 0, :, :]
    G = img[:, 1, :, :]
    B = img[:, 2, :, :]
    
    # 1. 计算对立颜色通道
    rg = R - G
    yb = 0.5 * (R + G) - B
    
    # 2. 在空间维度 (H, W) 上计算均值和方差，dim=(1, 2)
    # 注意：直接使用 var 而不是 std，因为公式中的 std**2 就是 var，这样能少算一次根号
    rg_mean = torch.mean(rg, dim=(1, 2))
    rg_var = torch.var(rg, dim=(1, 2), unbiased=False)
    
    yb_mean = torch.mean(yb, dim=(1, 2))
    yb_var = torch.var(yb, dim=(1, 2), unbiased=False)
    
    # 3. 组合指标 (极其关键：加入 1e-8 防止算 sqrt(0) 导致梯度 NaN 崩溃)
    std_root = torch.sqrt(rg_var + yb_var + 1e-8)
    mean_root = torch.sqrt(rg_mean**2 + yb_mean**2 + 1e-8)
    
    # 返回每个 Batch 的 colorfulness，形状为 [B]
    # 这里的 0.3 是原始公式固有的权重
    colorfulness = std_root + 0.3 * mean_root
    
    return colorfulness

def color_mse_loss(pred, y, mode='lab', color_weight=0.3):
    """
    结合 MSE 和 Colorfulness 的复合损失函数
    
    参数:
    pred: 模型预测图像 [B, 3, H, W], 取值 [0, 1]
    y: 真实图像 (Ground Truth) [B, 3, H, W], 取值 [0, 1]
    color_weight: 色彩丰富度损失的权重，默认 0.3
    """
    if mode == 'lab':
        mse = F.mse_loss(pred, y)
        pixels = pred.numel()
        pred_color = torch.sum((pred - 0.5) ** 2) / pixels
        y_color = torch.sum((y - 0.5) ** 2) / pixels
        color_diff = F.l1_loss(pred_color, y_color)

        loss = color_diff * color_weight + mse
        return loss
    elif mode == 'rgb':
    # 1. 基础 MSE Loss (计算像素级误差)
        mse = F.mse_loss(pred, y)
        
        # 2. 计算预测图和真实图的色彩丰富度
        c_pred = differentiable_colorfulness(pred)
        c_y = differentiable_colorfulness(y)
        
        # 3. Colorfulness 差异损失 (使用 L1 距离)
        # 目的：让模型生成的色彩丰富度贴合真实图片，惩罚“掉色”或“过度饱和”
        color_loss = F.l1_loss(c_pred, c_y)
        
        # 4. 复合总 Loss (你要求的 0.3 外部权重)
        total_loss = mse + color_weight * color_loss
        
        return total_loss