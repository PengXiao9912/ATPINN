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
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 设置随机种子以确保结果的可重复性
init_seed = 0
np.random.seed(init_seed)
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)

# 确保使用的是确定的算法，以便于结果重现
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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
            nn.Linear(hidden_size, output_size),
        )
        # 将 C_m, A_s, C_w 初始化为可训练的参数
        self.C_m = nn.Parameter(torch.randn(1))
        self.A_s = nn.Parameter(torch.randn(1))
        self.C_w = nn.Parameter(torch.randn(1))

    def forward(self, inputs):
        # inputs 应该是一个包含 t, omega, k, H 的张量
        attended_inputs = self.attention(inputs)
        return self.net(attended_inputs), self.C_m, self.A_s, self.C_w,

# 定义 Rumer 方程的残差计算
def compute_rumer_residual(model, inputs, params):
    prediction, C_m, A_s, C_w= model(inputs)
    U = prediction[:, 0:1]
    eta = prediction[:, 1:2]
    t = inputs[:, 0]
    omega = inputs[:, 1]
    k = inputs[:, 2]
    h = inputs[:, 3]
    x = inputs[:, 4]

    # 使用自动微分计算导数
    U_all = torch.autograd.grad(U, inputs, grad_outputs=torch.ones_like(U), create_graph=True)[0]
    eta_all = torch.autograd.grad(eta, inputs, grad_outputs=torch.ones_like(eta), create_graph=True)[0]
    # print(U_all.shape)
    U_t = U_all[:,0:1]
    eta_x = eta_all[:,4]
    # print(U_t.shape)

    # 提取物理参数
    m = params['m']
    g = params['g']
    rho = params['rho']

    # 计算 tanh(k*h) 需要使用 torch.tanh
    th_kh = torch.tanh(k * h)

    # m-浮冰质量、g-重力加速度、rho-水的密度、As-湿表面积、Cw-拖曳系数、Cm附加质量系数、omega-波浪圆频率、k-波数、H-波高、U-浮冰速度、eta-位移
    # 根据 Rumer 方程计算残差
    residual = m * (1 + C_m) * U_t + m * g * eta_x - \
               rho * A_s * C_w * torch.abs((omega * eta / th_kh) - U) * ((omega * eta / th_kh) - U)

    return residual.squeeze(1)

# 损失函数
def loss_function(model, inputs, U_data, params):
    prediction, C_m, A_s, C_w= model(inputs)
    U = prediction[:, 0:1]
    eta = prediction[:, 1:2]

    # 计算数据损失
    data_loss_U = nn.MSELoss()(U, U_data) 

    # 计算 Rumer 方程的残差
    # 这里我们将固定参数和学习参数一起传递给残差计算函数
    pde_loss = compute_rumer_residual(model, inputs, params).pow(2).mean()

    return data_loss_U + pde_loss 

def test_model(model, test_dataloader, optims: list):
    model.eval()
    total_test_loss = 0
    for data in test_dataloader:
        data = data[0]
        t_omega_k_H_x_data, U_data = data
        t_omega_k_H_x_data = t_omega_k_H_x_data.to(device)
        U_data = U_data.to(device)

        # 在这个块内部，临时开启梯度计算
        test_loss = loss_function(model, t_omega_k_H_x_data.requires_grad_(), U_data.unsqueeze(1), params)
        total_test_loss += test_loss.item()
    average_test_loss = total_test_loss / len(test_dataloader.dataset)
    for optim in optims:
        optim.zero_grad()
    model.train()
    return average_test_loss

# 实例化模型
input_size = 5  # 输入是时间、波浪频率、波数、波高、x
hidden_size = 256
output_size = 2  # 输出包括 U 和 eta
attention_size = 32

if __name__ == '__main__':

    model = PINNModel(input_size, hidden_size, output_size, attention_size).to(device)
    
    # 定义参数
    # 水池试验浮冰0.9325*0.6325*0.012；哈尔滨的重力加速度9.80665
    params = {"m": 4.3206075, "g": 9.80665, "rho": 1000.0, }
    
    # 优化器
    optimizer_adamw = optim.AdamW(model.parameters(), lr=2e-8)
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1e-8, max_iter=300)

    # 加载所有数据
    # 设置包含 CSV 文件的文件夹路径
    folder_path = '/root/Code/ice protection/train_data' 
    test_folder_path = '/root/Code/ice protection/test_data' 
    
    batch_size = 4000  # 定义批大小
    num_workers = 4
    pin_memory= True

    train_dataset = PINNsDataset(folder_path, batch_size)
    train_dataloader = DataLoader(train_dataset,
                                batch_size=1,
                                shuffle=True,
                                num_workers=num_workers,
                                pin_memory=pin_memory,
                                collate_fn=pinns_collect_fn)
    

    test_dataset = PINNsDataset(test_folder_path, batch_size)  # 使用测试数据路径
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=1, 
                                 shuffle=False, 
                                 num_workers=num_workers, 
                                 pin_memory=pin_memory, collate_fn=pinns_collect_fn)

 
    # 初始化日志字典
    # 初始化日志字典
    log = {
        'total_loss': [],
        'max_total_loss': [],
        'min_total_loss': [],
        'pde_loss': [],
        'data_loss_U': [],
        'test_loss': [],
    }

    epoch = 8000
    switch_to_lbfgs = 6000
    
    loss_func = nn.MSELoss()

    # 训练循环
    for i in range(epoch):
        # 初始化最大和最小总损失值
        total_losses, pde_losses, data_losses_U, test_losses = [], [], [], []
        block_total_losses, block_pde_losses, block_data_losses_U, block_test_losses = [], [], [], []

        with tqdm(train_dataloader, total=train_dataset.data_len, desc=f'Epoch {i+1}/{epoch}') as pbar:
            file_count = 0
            for data in pbar:
                t1 = time.time()

                data = data[0]
                t_omega_k_h_xdata, U_data = data
                t_omega_k_h_xdata = t_omega_k_h_xdata.requires_grad_()
                t_omega_k_h_xdata = t_omega_k_h_xdata.to(device)
                U_data = U_data.to(device)
            
                # 定义闭包函数用于优化
                def closure():
                    optimizer_lbfgs.zero_grad()
                    prediction, C_m, A_s, C_w= model(t_omega_k_h_xdata)
                    data_loss_U = loss_func(prediction[:, 0:1], U_data.unsqueeze(1)) 
                    pde_loss = compute_rumer_residual(model, t_omega_k_h_xdata, params).pow(2).mean()
                    total_loss = data_loss_U + pde_loss
                    total_loss.backward()
                    # for name, param in model.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"Gradient of {name}: {param.grad.norm()}")
                     # 梯度裁剪
                    clip_value = 100.0  # 设置一个阈值
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    return total_loss
        
                # 优化步骤
                if i < switch_to_lbfgs:
                    optimizer_adamw.zero_grad()
                    prediction, C_m, A_s, C_w= model(t_omega_k_h_xdata)
                    data_loss_U = loss_func(prediction[:, 0:1], U_data.unsqueeze(1))
                    pde_loss = compute_rumer_residual(model, t_omega_k_h_xdata, params).pow(2).mean()
                    total_loss = data_loss_U  + pde_loss
                    total_loss.backward()
                    # for name, param in model.named_parameters():
                    #     if param.grad is not None:
                    #         print(f"Gradient of {name}: {param.grad.norm()}")
                     # 梯度裁剪
                    clip_value = 100.0  # 设置一个阈值
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                    optimizer_adamw.step()
                else:
                    total_loss = optimizer_lbfgs.step(closure)
                    # 重新计算其他损失值
                    with torch.no_grad():
                        prediction, C_m, A_s, C_w= model(t_omega_k_h_xdata)
                        data_loss_U = loss_func(prediction[:, 0:1], U_data.unsqueeze(1)) 
                    # 重新计算 PDE 损失时需要梯度
                    t_omega_k_h_xdata_grad = t_omega_k_h_xdata.clone().requires_grad_()  # 重新设置 requires_grad
                    pde_loss = compute_rumer_residual(model, t_omega_k_h_xdata_grad, params).pow(2).mean()
    
                # 累积每个文件的误差
                total_losses.append(total_loss.item())
                pde_losses.append(pde_loss.item())
                data_losses_U.append(data_loss_U.item())
                file_count += 1
                
                # 每处理50个文件后，计算平均、最大和最小值，并更新日志
                if file_count % 44 == 0 or file_count == len(train_dataloader):
                    
                    avg_total_loss = sum(total_losses) / len(total_losses)
                    avg_pde_loss = sum(pde_losses) / len(pde_losses)
                    avg_data_loss_U = sum(data_losses_U) / len(data_losses_U)
                    max_total_loss = max(total_losses)
                    min_total_loss = min(total_losses)
    
                    log['total_loss'].append(avg_total_loss)
                    log['max_total_loss'].append(max_total_loss)
                    log['min_total_loss'].append(min_total_loss)
                    log['pde_loss'].append(avg_pde_loss)
                    log['data_loss_U'].append(avg_data_loss_U)

                    # 将每50个文件块的平均损失存储起来
                    block_total_losses.append(avg_total_loss)
                    block_pde_losses.append(avg_pde_loss)
                    block_data_losses_U.append(avg_data_loss_U)

                    # 更新进度条以显示当前50个文件块的平均损失
                    pbar.set_postfix({
                        'Avg Total Loss': avg_total_loss,
                        'Avg PDE Loss': avg_pde_loss,
                        'Avg Data Loss U': avg_data_loss_U,
                        # 'Avg Test Loss': avg_test_loss
                    })

                    # 重置累积变量
                    total_losses, pde_losses, data_losses_U, test_losses = [], [], [], []

                t2 = time.time()  # 结束时间
                pbar.update()

        # 每次迭代结束后计算并打印整个迭代的平均损失
        avg_total_loss = sum(block_total_losses) / len(block_total_losses) if block_total_losses else 0
        avg_pde_loss = sum(block_pde_losses) / len(block_pde_losses) if block_pde_losses else 0
        avg_data_loss_U = sum(block_data_losses_U) / len(block_data_losses_U) if block_data_losses_U else 0
        # avg_test_loss = sum(block_test_losses) / len(block_test_losses) if block_test_losses else 0

        avg_test_loss = test_model(model, test_dataloader, [optimizer_adamw, optimizer_lbfgs])
        log['test_loss'].append(avg_test_loss)
        block_test_losses.append(avg_test_loss)
    
        print(f'Epoch {i+1} - Avg Total Loss: {avg_total_loss}, Avg PDE Loss: {avg_pde_loss}, '
              f'Avg Data Loss U: {avg_data_loss_U}', f'Avg Test Loss: {avg_test_loss}')
    
    
        # 每1000次迭代保存一次模型
        if (i + 1) % 2000 == 0:
            torch.save(model.state_dict(), f'/root/Code/ice protection/20240418log/4-256_model_20240411{i+1}.pth')
        

    for key in log.keys():
        print(f"Length of '{key}': {len(log[key])}")

    log_df = pd.DataFrame(log)
    log_df.to_csv('/root/Code/ice protection/20240418log/4-256training_log20240411.csv', index=False)