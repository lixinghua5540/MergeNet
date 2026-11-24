import torch
import torch.nn as nn
try:
    import timm
    from timm.utils import flops_counter
except ImportError:
    print("请先安装 timm 库: pip install timm")

def check_backbone_stats(model_name, input_size=(1, 3, 512, 512)):
    print(f"正在分析模型: {model_name} ...")
    
    # 1. 创建模型 (num_classes=0 表示只取特征，不包含最后的分类层)
    # features_only=True 有时用于获取多尺度中间特征，这里我们取全局池化后的特征向量
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    model.eval()

    # 2. 计算参数量 (Parameters)
    params = sum(p.numel() for p in model.parameters()) / 1e6

    # 3. 计算计算量 (FLOPs) - 使用 timm 自带工具
    dummy_input = torch.randn(*input_size)
    try:
        from fvcore.nn import FlopCountAnalysis
        flops_analyser = FlopCountAnalysis(model, dummy_input)
        flops = flops_analyser.total() / 1e9
    except ImportError:
        # 如果没装 fvcore，用简单粗暴的估计
        flops = "需安装 fvcore 获取精确值"

    print(f"  - 参数量 (Params): {params:.2f} M")
    print(f"  - 计算量 (FLOPs):  {flops:.2f} G (输入 {input_size[2]}x{input_size[3]})")
    print("-" * 40)

if __name__ == "__main__":
    # 定义你想查看的模型列表 (timm 中的标准命名)
    backbones = [
        'resnext50_32x4d', 
        'res2net50_26w_4s', 
        'efficientnet_b0', 
        'pvt_v2_b2'
    ]

    for name in backbones:
        try:
            check_backbone_stats(name)
        except Exception as e:
            print(f"无法加载模型 {name}: {e}")

    print("\n[注]: PVTv2 在 timm 中建议使用 pvt_v2_b2 等具体规格。")