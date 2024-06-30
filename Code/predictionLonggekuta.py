import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from PinnDataSet import PINNsDataset, pinns_collect_fn
from torch.utils.data.dataloader import DataLoader

# 检查CUDA（GPU支持）是否可用，否则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 设置随机种子以确保结果的可重复性
init_seed = 0
np.random.seed(init_seed)
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)

# 确保使用的是确定的算法，以便于结果重现
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

    
# 定义神经网络模型
class Attention(nn.Module):
    def __init__(self, input_size, attention_size):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1)
        )
     
    def forward(self, x):
        # x shape: [batch_size, 4] assuming x includes [t, wave_info]
        attention_weights = torch.softmax(self.attention(x), dim=1)
        attended_x = attention_weights * x
        return attended_x
    
# 定义神经网络模型

class Attention(nn.Module):
    def __init__(self, input_size, attention_size):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1)
        )
     
    def forward(self, x):
        # x shape: [batch_size, 4] assuming x includes [t, wave_info]
        attention_weights = torch.softmax(self.attention(x), dim=1)
        attended_x = attention_weights * x
        return attended_x

class PINNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, attention_size):
        super(PINNModel, self).__init__()
        self.attention = Attention(input_size, attention_size)
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )
    def forward(self, inputs):
        # inputs 应该是一个包含 t, omega, k, H 的张量
        attended_inputs = self.attention(inputs)
        return self.net(attended_inputs)

# 指定模型和数据路径
model_path = "/root/Code/ice protection/20240418log/MLP-attention_model_epoch_8000.pth"
input_csv_path = "/root/Code/ice protection/test_data/T=1.2-Ice5.CSV"
output_csv_path = "/root/Code/ice protection/prediction/T=1.2-ModelMLP-attention.CSV"

# 加载模型
input_size = 5  # 根据你的模型实际输入调整
hidden_size = 256  # 根据你的模型实际设置调整
output_size = 2  # 根据你的模型实际输出调整
attention_size = 32
model = PINNModel(input_size, hidden_size, output_size, attention_size)
model.load_state_dict(torch.load(model_path))
model.eval()

# 检查设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 读取输入数据
data_df = pd.read_csv(input_csv_path, encoding='gbk', usecols=[0, 1, 2, 3, 4])
t_values = data_df.iloc[:, 0].to_numpy(dtype=np.float32)  # Assuming the first column is time 't'
print(t_values)
inputs_np = data_df.to_numpy(dtype=np.float32)
inputs_tensor = torch.from_numpy(inputs_np)
inputs_tensor = inputs_tensor.to(device)

# 计算时间间隔 dt
dt_values = np.diff(t_values, prepend=t_values[0])  # Compute time differences

# 进行预测
with torch.no_grad():
    predictions = model(inputs_tensor)
predictions_np = predictions.cpu().numpy()

def runge_kutta_4(model, t, x, dt, omega, k, h):
    """Compute next displacement using Runge-Kutta 4th order method."""
    def velocity(t, x):
        inputs = torch.tensor([t, omega, k, h, x], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            return outputs[:, 0:1].item()  # Assuming the second output is velocity

    k1 = dt * velocity(t, x)
    k2 = dt * velocity(t + 0.5 * dt, x + 0.5 * k1)
    k3 = dt * velocity(t + 0.5 * dt, x + 0.5 * k2)
    k4 = dt * velocity(t + dt, x + k3)
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6

# 初始化位移
x = 0.0  # 初始位移，根据需要进行调整
displacements = []

# 循环通过每个时间点进行积分
for i in range(len(inputs_np)):
    t, omega, k, h, _ = inputs_np[i]
    dt = dt_values[i]  # Use computed time difference
    x = runge_kutta_4(model, t, x, dt, omega, k, h)
    displacements.append(x)

# 将位移添加到结果 DataFrame
output_df = pd.DataFrame(predictions_np, columns=["U_pred", "eta_pred"])
output_df['Displacement'] = displacements
output_df.to_csv(output_csv_path, index=False)
print(f"Updated predictions with displacements have been saved to {output_csv_path}")