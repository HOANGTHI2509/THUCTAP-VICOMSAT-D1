import sys
import os
import torch
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.tcn_model import FuelCausalTCN

def verify_receptive_field_and_causality():
    model = FuelCausalTCN()
    model.eval()
    
    # We create a dummy input of shape (Batch=1, SeqLen=30, Features=5)
    # Require gradients to calculate Jacobian/Gradients wrt input
    x = torch.zeros(1, 30, 5, requires_grad=True)
    
    # Forward pass
    out = model(x) # Output is scalar for batch 1
    
    # Backpropagate to see which input elements have non-zero gradients
    out.backward()
    
    # Gradients of output wrt input x
    grads = x.grad.abs().sum(dim=(0, 2)) # Sum over Batch and Feature dims, leaving SeqLen
    
    print("=== BÁO CÁO KIỂM ĐỊNH CAUSALITY VÀ RECEPTIVE FIELD ===")
    
    non_zero_grads = torch.where(grads > 0)[0]
    
    if len(non_zero_grads) == 0:
        print("Lỗi: Gradient bằng 0, mô hình không học được gì.")
        return
        
    first_influenced_idx = non_zero_grads[0].item()
    last_influenced_idx = non_zero_grads[-1].item()
    
    print(f"Timestep cuối cùng được sử dụng (t_current): {last_influenced_idx}")
    print(f"Timestep xa nhất trong quá khứ ảnh hưởng đến output: {first_influenced_idx}")
    receptive_field = last_influenced_idx - first_influenced_idx + 1
    print(f"Receptive Field thực tế: {receptive_field} timesteps")
    
    # Kiểm tra Causality
    # Output tại t=29. Nếu nó dùng thông tin từ tương lai thì max idx sẽ > 29.
    # Tuy nhiên vì input chỉ dài 30, t=29 là điểm cuối.
    # Sẽ kiểm tra thêm 1 bài toán mô phỏng Causal: 
    # Đổi dữ liệu của các index sau t_current xem output tại t_current có đổi không?
    # Vì output của chúng ta lấy x_last = x[:, :, -1], tức là t=29, ta không có tương lai.
    # Nhưng ta có thể lấy timestep t=20 ra làm output để test!
    
    print("\n--- KIỂM TRA CAUSALITY SÂU HƠN ---")
    
    # Customize the model to output at t=20 instead of t=29
    def forward_at_t(self, x_in, t_target=20):
        x_trans = x_in.transpose(1, 2)
        h = self.relu1(self.conv1(x_trans))
        h = self.relu2(self.conv2(h))
        h = self.relu3(self.conv3(h))
        x_t = h[:, :, t_target] # Lấy tại t=20
        out = self.relu4(self.dense1(x_t))
        return self.dense2(out).squeeze(-1)
        
    # Bind the custom method temporarily
    import types
    model.forward_at_t = types.MethodType(forward_at_t, model)
    
    x2 = torch.randn(1, 30, 5)
    out1 = model.forward_at_t(x2, t_target=20)
    
    # Thay đổi toàn bộ dữ liệu ở tương lai (t=21 đến 29)
    x3 = x2.clone()
    x3[:, 21:, :] = torch.randn(1, 9, 5) * 1000 # Thay đổi cực lớn
    
    out2 = model.forward_at_t(x3, t_target=20)
    
    diff = torch.abs(out1 - out2).item()
    print(f"Sự khác biệt output tại t=20 khi thay đổi toàn bộ dữ liệu tương lai t>20: {diff:.10f}")
    if diff < 1e-6:
        print("✅ CAUSALITY PASS: Output tại thời điểm t hoàn toàn KHÔNG phụ thuộc vào dữ liệu tại t+1, t+2...")
    else:
        print("❌ CAUSALITY FAIL: Mô hình bị rò rỉ dữ liệu tương lai (Leaky).")
        
    print("\n--- GIẢI THÍCH VỀ RECEPTIVE FIELD 15 TIMESTEPS ---")
    explanation = (
        "Receptive field của kiến trúc được tính như sau:\n"
        "- Lớp Conv1 (dilation=1): Nhìn lùi 2 điểm.\n"
        "- Lớp Conv2 (dilation=2): Nhìn lùi 4 điểm.\n"
        "- Lớp Conv3 (dilation=4): Nhìn lùi 8 điểm.\n"
        "-> Tổng cộng: 2 + 4 + 8 = 14 điểm quá khứ + 1 điểm hiện tại = 15 điểm.\n\n"
        "Mặc dù Input Window có kích thước 30 điểm (30 phút), nhưng TCN chỉ sử dụng "
        "đúng 15 điểm gần nhất để dự báo. ĐÂY LÀ CHỦ Ý THIẾT KẾ, KHÔNG PHẢI LỖI, VÌ:\n"
        "1. 30 điểm ban đầu được truyền vào mục đích chính là để cung cấp đủ số liệu cho hàm tính Anchor (Rolling Median) chạy ngoài Data Pipeline.\n"
        "2. Về mặt vật lý, biến động bất thường của xăng (Rút/Đổ/Sloshing) diễn ra trong một khoảng thời gian ngắn. Receptive field 15 phút là một khoảng thời gian "
        "vừa đủ (ngắn hạn) để TCN nhận diện sự kiện, giúp tránh Overfitting vào các hành vi lái xe dài hạn không liên quan."
    )
    print(explanation)

if __name__ == '__main__':
    verify_receptive_field_and_causality()
