import matplotlib.pyplot as plt

# 输入文件名
load_path = "./clients/" + "2023-06-01-12_52_32/"
file_name = "2023-06-01-12_52_32_client_1.log"
# print(file_name.split("---"))
save_path = "./image/"

# 读取输入文件
with open(load_path + file_name, 'r') as f:
    lines = f.readlines()

# 提取 HR 和 NDCG
epochs = []
train_loss_list = []
HR_list = []
NDCG_list = []

# 提取 loss 记载起始行
start_line = 0
for i in range(len(lines)):
    if lines[i].startswith('epoch-'):
        start_line = i
        break

# print(start_line)
# 提取数据
for i in range(start_line, len(lines)-1, 5):
    line2 = lines[i+1].strip().split()  # 处理第二行

    epochs.append(int(line2[3].strip(',')))
    train_loss_list.append(float(line2[6].strip(',')))
    HR_list.append(float(line2[8].strip(',')))
    NDCG_list.append(float(line2[10]))
    # # 执行所需的操作
    # if line.startswith('说明：'):
    #     parts = line.split(',')
    #     x = parts[0][3:]
    #     y = parts[1][3:-1]
    #     xs.append(float(x))
    #     ys.append(float(y))

# print(epochs)
# print(train_loss_list)
# print(HR_list)
# print(NDCG_list)

# 绘制图表
plt.figure()
plt.plot(epochs, train_loss_list)
plt.xlabel('epoch')
plt.ylabel('train_loss')
plt.title('epochs vs train_loss')
plt.savefig(save_path + 'train_loss-' + file_name + '.png')

plt.figure()
plt.plot(epochs, HR_list)
plt.xlabel('epoch')
plt.ylabel('HR')
plt.title('epochs vs HR')
plt.savefig(save_path + 'HR-' + file_name + '.png')

plt.figure()
plt.plot(epochs, NDCG_list)
plt.xlabel('epoch')
plt.ylabel('NDCG')
plt.title('epochs vs NDCG')
plt.savefig(save_path + 'NDCG-' + file_name + '.png')
plt.show()