import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    """
    Grad-CAM cho backbone CNN (EfficientNet) trong mô hình EfficientNet + LSTM.
    target_layer: layer CNN muốn lấy activation (thường là backbone._conv_head hoặc _blocks[-3]).
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Hook forward lấy feature map
        target_layer.register_forward_hook(self._forward_hook)

        # Hook backward lấy gradient
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        # KHÔNG detach ở đây để gradient còn chảy
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, class_score):
        """
        Trả về CAM shape: (B*T, 1, H_feat, W_feat)
        """
        grads = self.gradients      # (B*T, C, H, W)
        acts = self.activations     # (B*T, C, H, W)

        # Grad-CAM step
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B*T, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True) # (B*T, 1, H, W)
        cam = F.relu(cam)

        # normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-7)

        return cam.detach().cpu()


def overlay_heatmap(frame, cam, alpha=0.7):
    """
    frame: H x W x 3 (uint8) - ảnh RGB
    cam:   H x W (float 0-1)
    """
    cam = cv2.resize(cam, (frame.shape[1], frame.shape[0]))

    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    result = cv2.addWeighted(heatmap, alpha, frame, 1 - alpha, 0)
    return result


def compute_gradcam(model, cam_extractor, video, target_class=None):
    """
    model: EfficientNet-LSTM
    cam_extractor: GradCAM instance
    video: tensor (1, 3, T, H, W)
    target_class: 
        - None: dùng class mà model dự đoán
        - 0: heatmap class Real
        - 1: heatmap class Fake

    Return:
    - cam: (1, T, 1, H_feat, W_feat)
    - predicted_class
    """

    device = next(model.parameters()).device
    video = video.to(device)
    video.requires_grad = True

    # 1. Forward
    output = model(video)  # (1,2)

    # 2. Chọn class để giải thích
    if target_class is None:
        class_id = output.argmax(dim=1).item()   # lấy class dự đoán
    else:
        class_id = int(target_class)

    # 3. Score cho class đích
    score = output[:, class_id]

    # 4. Backward
    model.zero_grad()
    score.backward(retain_graph=True)

    # 5. Tạo CAM
    cam = cam_extractor.generate_cam(score)   # (B*T,1,h,w)

    B, C, T, H, W = video.shape
    cam = cam.reshape(B, T, 1, cam.shape[2], cam.shape[3])  # -> (B,T,1,h,w)

    return cam, class_id