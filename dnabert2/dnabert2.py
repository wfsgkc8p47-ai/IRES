import pandas as pd
import numpy as np
import torch
import os
from transformers import AutoModel, AutoTokenizer

# 指定本地路径
local_model_path = r"D:\学习\新的\IRESfinder-master\dnabert2_local"

# 从本地加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModel.from_pretrained(local_model_path)

print("DNABERT-2 已成功从本地加载！")

# 读取 CSV 文件
file_path = r"D:\学习\新的\IRESfinder-master\IRESfinder-master\data2\train_0_part_9.csv"  # 确保 CSV 文件已上传
df = pd.read_csv(file_path)

# 确保 CSV 里有 "sequence" 列
if 'seq' not in df.columns:
    raise ValueError("CSV 文件缺少 'seq' 列，请检查数据！")

# 计算嵌入
model.eval()  # 设置为评估模式
embeddings = []

for seq in df['seq']:
    inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():  # 关闭梯度计算
        outputs = model(**inputs)

    # 取 [CLS] token 的嵌入
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    embeddings.append(cls_embedding)

# 转换为 NumPy 数组并保存
embeddings = np.array(embeddings)
output_dir = r"D:\学习\新的\IRESfinder-master\dnabert2\d2"
output_file = os.path.join(output_dir, "embeddingstr9.npy")
np.save(output_file, embeddings)

print(f"嵌入计算完成，已保存到 {output_file}")
