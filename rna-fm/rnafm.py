import fm
import torch
import numpy as np
import pandas as pd

# 加载 RNA-FM 预训练模型
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 读取 CSV 文件
csv_file = r"D:\学习\新的\IRESfinder-master\IRESfinder-master\data\5utr_converted_uppercase.csv"
df = pd.read_csv(csv_file)

# 分批处理参数
batch_size = 1
num_sequences = len(df)
all_embeddings = []

# 分批处理
for start_idx in range(0, num_sequences, batch_size):
    end_idx = min(start_idx + batch_size, num_sequences)
    batch_df = df.iloc[start_idx:end_idx]
    data = list(zip(batch_df.index.astype(str), batch_df["seq"]))

    # 转换为 tokens
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    # 提取特征
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[12])
        token_embeddings = results["representations"][12]

    # 计算平均特征
    sequence_embeddings = token_embeddings.mean(dim=1)
    all_embeddings.append(sequence_embeddings.cpu().numpy())

    print(f"Processed sequences {start_idx} to {end_idx - 1}")

# 合并所有批次的特征
sequence_embeddings_np = np.concatenate(all_embeddings, axis=0)
np.save("sequence_embeddings 5utr.npy", sequence_embeddings_np)

print(f"Sequence embeddings for {num_sequences} sequences saved to 'sequence_embeddings.npy'")
# import fm
# import torch
# import numpy as np
# import pandas as pd
#
# # 加载 RNA-FM 预训练模型
# model, alphabet = fm.pretrained.rna_fm_t12()
# batch_converter = alphabet.get_batch_converter()
#
# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
#
# # 读取 CSV 文件
# csv_file = r"D:\学习\新的\IRESfinder-master\IRESfinder-master\data\5utr_converted_uppercase.csv"
# df = pd.read_csv(csv_file)
#
# # 分批处理参数
# batch_size = 50  # 每次处理50条序列，可根据内存调整
# max_length = 1024  # 模型支持的最大序列长度（根据 RNA-FM 的限制）
# window_size = 512  # 滑动窗口大小（可以调整）
# window_step = 256  # 滑动窗口步长（可以调整）
# all_embeddings = []
#
# # 分批处理
# for start_idx in range(0, len(df), batch_size):
#     end_idx = min(start_idx + batch_size, len(df))
#     batch_df = df.iloc[start_idx:end_idx]
#
#     # 处理每个序列
#     for seq_idx, row in batch_df.iterrows():
#         sequence = row["seq"]
#         sequence_length = len(sequence)
#
#         # 如果序列长度超过模型支持的最大长度，使用滑动窗口分割
#         if sequence_length > max_length:
#             # 滑动窗口分割
#             window_embeddings = []
#             for window_start in range(0, sequence_length, window_step):
#                 window_end = min(window_start + window_size, sequence_length)
#                 window_sequence = sequence[window_start:window_end]
#
#                 # 转换为 tokens
#                 data = [(str(seq_idx) + f"_window_{window_start}", window_sequence)]
#                 batch_labels, batch_strs, batch_tokens = batch_converter(data)
#                 batch_tokens = batch_tokens.to(device)
#
#                 # 提取特征
#                 with torch.no_grad():
#                     results = model(batch_tokens, repr_layers=[12])
#                     token_embeddings = results["representations"][12]
#
#                 # 计算窗口特征
#                 window_embedding = token_embeddings.mean(dim=1).cpu().numpy()
#                 window_embeddings.append(window_embedding)
#
#             # 合并窗口特征（可以使用平均或其他方法）
#             sequence_embedding = np.mean(window_embeddings, axis=0)
#         else:
#             # 直接处理短序列
#             data = [(str(seq_idx), sequence)]
#             batch_labels, batch_strs, batch_tokens = batch_converter(data)
#             batch_tokens = batch_tokens.to(device)
#
#             # 提取特征
#             with torch.no_grad():
#                 results = model(batch_tokens, repr_layers=[12])
#                 token_embeddings = results["representations"][12]
#
#             # 计算平均特征
#             sequence_embedding = token_embeddings.mean(dim=1).cpu().numpy()
#
#         all_embeddings.append(sequence_embedding)
#
#         # 打印处理进度
#         print(f"Processed sequence {seq_idx} (length: {len(sequence)})")
#
# # 转换为 NumPy 数组
# sequence_embeddings_np = np.array(all_embeddings)
# np.save("sequence_embeddings 5utr.npy", sequence_embeddings_np)
#
# print(f"Sequence embeddings for {len(df)} sequences saved to 'sequence_embeddings_main_independent.npy'")