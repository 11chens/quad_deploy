import torch
import os
import onnx
import onnxruntime as ort
import numpy as np
import time


def convert_jit_to_onnx(model, dummy_input, output_path, input_names=None, output_names=None):
    """
    将JIT模型转换为ONNX格式
    
    参数：
        model: 加载好的JIT模型
        dummy_input: 示例输入张量
        output_path: 输出的ONNX文件路径
        input_names: 输入节点名称列表，默认["input"]
        output_names: 输出节点名称列表，默认["output"]
    """
    # 确保模型在CPU模式（ONNX导出推荐使用CPU）
    model = model.cpu()
    dummy_input = dummy_input.cpu()

    # 设置默认输入输出名称
    input_names = input_names or ["input"]
    output_names = output_names or ["output"]
    try:
        # 导出为ONNX
        torch.onnx.export(model, dummy_input, output_path, verbose=False, input_names=input_names,
                          output_names=output_names)
        print(f"成功导出ONNX到 {output_path}")
    except Exception as e:
        print(f"导出失败: {str(e)}")
        raise


if __name__ == "__main__":
    device = "cpu"
    nav_model_dir = "example/models/503"
    loco_model_dir = "example/python/models/403"
    loco_output_dir = "example/quad_deploy/models/onnx_models/locomotion_model"
    nav_output_dir = "example/quad_deploy/models/onnx_models/navigation_model"
    os.makedirs(loco_output_dir, exist_ok=True)
    os.makedirs(nav_output_dir, exist_ok=True)

    # nav_models = {
    #     "policy": torch.jit.load(os.path.join(nav_model_dir, 'policy.pt')).to(device),
    #     "encoder_rays": torch.jit.load(os.path.join(nav_model_dir, 'encoder_rays.pt')).to(device),
    #     "encoder_prop": torch.jit.load(os.path.join(nav_model_dir, 'encoder_prop.pt')).to(device)
    # }

    # for name, model in nav_models.items():
    #     dummy_input = torch.randn(77 if name == "policy" else 155 if name == "encoder_rays" else 120, device=device)
    #     # dummy_input = torch.randn(49 if name == "policy" else 120, device=device)
    #     # dummy_input = torch.randn(93, device=device)
    #     output_path = os.path.join(nav_output_dir, f"{name}.onnx")
    #     convert_jit_to_onnx(model, dummy_input, output_path)

    loco_models = {
        "body_latest": torch.jit.load(os.path.join(loco_model_dir, 'body_latest.jit')).to(device),
        "encoder_vel": torch.jit.load(os.path.join(loco_model_dir, 'encoder_vel.jit')).to(device)
    }

    for name, model in loco_models.items():
        dummy_input = torch.randn(48 if name == "body_latest" else 450, device=device)
        output_path = os.path.join(loco_output_dir, f"{name}.onnx")
        convert_jit_to_onnx(model, dummy_input, output_path)
