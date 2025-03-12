import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# 定义一个贝叶斯线性层
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 权重的先验分布参数
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.ones(out_features, in_features) * -3)
        
        # 偏置的先验分布参数
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.ones(out_features) * -3)
        
        # 初始化标准正态分布
        self.weight_prior = Normal(0, 1)
        self.bias_prior = Normal(0, 1)

    def forward(self, x):
        # 从权重分布中采样
        weight_sigma = F.softplus(self.weight_rho)
        weight_dist = Normal(self.weight_mu, weight_sigma)
        weight_sample = weight_dist.rsample()
        
        # 从偏置分布中采样
        bias_sigma = F.softplus(self.bias_rho)
        bias_dist = Normal(self.bias_mu, bias_sigma)
        bias_sample = bias_dist.rsample()
        
        # 计算前向传播
        return F.linear(x, weight_sample, bias_sample)
    
    def kl_divergence(self):
        # KL 散度计算
        weight_sigma = F.softplus(self.weight_rho)
        weight_dist = Normal(self.weight_mu, weight_sigma)
        kl_weight = torch.sum(torch.distributions.kl_divergence(weight_dist, self.weight_prior))
        
        bias_sigma = F.softplus(self.bias_rho)
        bias_dist = Normal(self.bias_mu, bias_sigma)
        kl_bias = torch.sum(torch.distributions.kl_divergence(bias_dist, self.bias_prior))
        
        return kl_weight + kl_bias

# 定义贝叶斯神经网络
class BayesianNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BayesianNN, self).__init__()
        self.bayes_fc1 = BayesianLinear(input_dim, hidden_dim)
        self.bayes_fc2 = BayesianLinear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.bayes_fc1(x))
        x = self.bayes_fc2(x)
        return x
    
    def kl_divergence(self):
        return self.bayes_fc1.kl_divergence() + self.bayes_fc2.kl_divergence()

# 超参数
input_dim = 1
hidden_dim = 16
output_dim = 1
epochs = 100
learning_rate = 0.01

# 数据生成
x = torch.linspace(-3, 3, 10).unsqueeze(1)
y = torch.sin(x) + 0.1 * torch.randn_like(x)

# 模型初始化
model = BayesianNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 损失函数
def loss_fn(pred, target, kl_divergence, beta=1.0):
    mse_loss = F.mse_loss(pred, target)
    return mse_loss + beta * kl_divergence

# 训练循环
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    y_pred = model(x)
    kl_div = model.kl_divergence()
    loss = loss_fn(y_pred, y, kl_div)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, KL Div: {kl_div.item():.4f}")

# 测试和预测
model.eval()
x_test = torch.linspace(-3, 3, 100).unsqueeze(1)
with torch.no_grad():
    y_test_pred = model(x_test)

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(x.numpy(), y.numpy(), label="Data", alpha=0.7)
plt.plot(x_test.numpy(), y_test_pred.numpy(), label="BNN Prediction", color="red")
plt.legend()
plt.show()
