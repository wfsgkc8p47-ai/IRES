import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# One-Hot 和 NCP 编码函数（适配 RNA）
def to_one_hot(sequence, max_length):
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
    seq_len = len(sequence)
    encoding = np.zeros((max_length, 4))
    for i in range(min(seq_len, max_length)):
        encoding[i] = mapping.get(sequence[i], [0, 0, 0, 0])
    return encoding

def to_properties_code(sequence, max_length):
    mapping = {
        'A': [1, 1, 1],  # 嘌呤, 2氢键, 弱
        'T': [0, 0, 1],  # 嘧啶, 2氢键, 弱
        'G': [1, 0, 0],  # 嘌呤, 3氢键, 强
        'C': [0, 1, 0]   # 嘧啶, 3氢键, 强
    }
    seq_len = len(sequence)
    encoding = np.zeros((max_length, 3))
    for i in range(min(seq_len, max_length)):
        encoding[i] = mapping.get(sequence[i], [0, 0, 0])
    return encoding


class SimpleEnhancer(nn.Module):
    def __init__(self, input_shape, fm_embedding_dim, dnabert_embedding_dim, num_classes=1, alpha=0.01):
        super(SimpleEnhancer, self).__init__()
        self.alpha = alpha  # 固定 alpha 值

        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        self.fm_weight = nn.Parameter(torch.tensor(1.0))
        self.dnabert_weight = nn.Parameter(torch.tensor(1.0))

        combined_dim = 128 + fm_embedding_dim + dnabert_embedding_dim
        self.fc1 = nn.Linear(combined_dim, 32)
        self.bn_fc = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, seq_data, fm_embedding, dnabert_embedding):
        x = self.conv1(seq_data)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = x.permute(0, 2, 1)
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)

        # 通过 alpha 调整 DNABERT 和 RNA-FM 的相对权重
        fm_embedding = (1 - self.alpha) * (self.fm_weight * fm_embedding)
        dnabert_embedding = self.alpha * (self.dnabert_weight * dnabert_embedding)

        combined = torch.cat([x, fm_embedding, dnabert_embedding], dim=1)
        combined = self.fc1(combined)
        combined = self.bn_fc(combined)
        combined = self.relu(combined)
        combined = self.dropout(combined)
        logits = self.fc2(combined)
        return logits


def build_improved_model(input_shape, fm_embedding_dim, dnabert_embedding_dim, pos_weight_value=1.0, device='cpu'):
    model = SimpleEnhancer(input_shape=input_shape, fm_embedding_dim=fm_embedding_dim,
                           dnabert_embedding_dim=dnabert_embedding_dim, num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    return model, optimizer, loss_fn, scheduler

def show_performance(y_true, y_pred, threshold=0.5):
    y_true = torch.as_tensor(y_true, dtype=torch.float32).flatten()
    y_pred = torch.as_tensor(y_pred, dtype=torch.float32).flatten()
    pred_binary = (y_pred > threshold).float()
    TP = torch.sum((y_true == 1) & (pred_binary == 1)).item()
    FN = torch.sum((y_true == 1) & (pred_binary == 0)).item()
    FP = torch.sum((y_true == 0) & (pred_binary == 1)).item()
    TN = torch.sum((y_true == 0) & (pred_binary == 0)).item()
    epsilon = 1e-6
    Sn = TP / (TP + FN + epsilon)
    Sp = TN / (FP + TN + epsilon)
    Acc = (TP + TN) / len(y_true)
    numerator = (TP * TN) - (FP * FN)
    denominator = torch.sqrt(torch.tensor((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) + epsilon))
    MCC = numerator / denominator
    return Sn, Sp, Acc, MCC

def performance_mean(performance):
    print('Sn = %.4f ± %.4f' % (np.mean(performance[:, 0]), np.std(performance[:, 0])))
    print('Sp = %.4f ± %.4f' % (np.mean(performance[:, 1]), np.std(performance[:, 1])))
    print('Acc = %.4f ± %.4f' % (np.mean(performance[:, 2]), np.std(performance[:, 2])))
    print('Mcc = %.4f ± %.4f' % (np.mean(performance[:, 3]), np.std(performance[:, 3])))
    print('Auc = %.4f ± %.4f' % (np.mean(performance[:, 4]), np.std(performance[:, 4])))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 读取训练和测试数据
    train_df = pd.read_csv(r'D:\学习\新的\IRESfinder-master\IRESfinder-master\dataset1\train.csv')
    test_df = pd.read_csv(r'D:\学习\新的\IRESfinder-master\IRESfinder-master\dataset1\test.csv')
    train_seqs = train_df['seq'].values
    train_labels = train_df['label'].values
    test_seqs = test_df['seq'].values
    test_labels = test_df['label'].values

    # 确定最大序列长度
    max_length = max(max(len(seq) for seq in train_seqs), max(len(seq) for seq in test_seqs))

    # 生成 One-Hot 和 NCP 编码
    train_onehot = np.array([to_one_hot(seq, max_length) for seq in train_seqs], dtype=np.float32)
    train_properties_code = np.array([to_properties_code(seq, max_length) for seq in train_seqs], dtype=np.float32)
    train_seq_data = np.concatenate((train_onehot, train_properties_code), axis=-1)  # (样本数, max_length, 7)

    test_onehot = np.array([to_one_hot(seq, max_length) for seq in test_seqs], dtype=np.float32)
    test_properties_code = np.array([to_properties_code(seq, max_length) for seq in test_seqs], dtype=np.float32)
    test_seq_data = np.concatenate((test_onehot, test_properties_code), axis=-1)  # (样本数, max_length, 7)

    # 加载 RNA-FM 嵌入（假设已预先计算并保存为 .npy 文件）
    train_fm_embeddings = np.load("rna-fm/sequence_embeddings_tr.npy")
    test_fm_embeddings = np.load("rna-fm/sequence_embeddings_t.npy")

    # 加载 DNABERT_2 嵌入（假设已预先计算并保存为 .npy 文件）
    train_db_embeddings = np.load("dnabert2/embeddings_tr.npy")
    test_db_embeddings = np.load("dnabert2/embeddings_t.npy")

    # 转换为 PyTorch 张量并移动到 device
    train_seq_data = torch.from_numpy(train_seq_data).to(device).permute(0, 2, 1)  # (样本数, 7, max_length)
    train_fm_embeddings = torch.from_numpy(train_fm_embeddings).to(device)          # (训练样本数, 640)
    train_dnabert_embeddings = torch.from_numpy(train_db_embeddings).to(device)      # (训练样本数, 768)
    train_labels = torch.from_numpy(train_labels).to(device).float()

    test_seq_data = torch.from_numpy(test_seq_data).to(device).permute(0, 2, 1)      # (样本数, 7, max_length)
    test_fm_embeddings = torch.from_numpy(test_fm_embeddings).to(device)            # (测试样本数, 640)
    test_dnabert_embeddings = torch.from_numpy(test_db_embeddings).to(device)        # (测试样本数, 768)
    test_labels = torch.from_numpy(test_labels).to(device).float()

    # 10折交叉验证
    n_splits = 10
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=1337)

    sv_10_result = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    # 计算正负样本比例，用于设置 pos_weight
    pos_count = np.sum(train_labels.cpu().numpy())
    neg_count = len(train_labels) - pos_count
    pos_weight_value = neg_count / pos_count

    for cycle in range(2):
        print('*' * 30 + f' Cycle {cycle + 1} ' + '*' * 30)
        all_Sn, all_Sp, all_Acc, all_MCC, all_AUC = [], [], [], [], []
        test_pred_all = []

        for fold_count, (train_index, val_index) in enumerate(k_fold.split(train_seq_data)):
            print('*' * 30 + f' Fold {fold_count + 1} ' + '*' * 30)
            tra_seq, val_seq = train_seq_data[train_index], train_seq_data[val_index]
            tra_fm, val_fm = train_fm_embeddings[train_index], train_fm_embeddings[val_index]
            tra_dnabert, val_dnabert = train_dnabert_embeddings[train_index], train_dnabert_embeddings[val_index]
            tra_labels, val_labels = train_labels[train_index], train_labels[val_index]

            train_dataset = TensorDataset(tra_seq, tra_fm, tra_dnabert, tra_labels)
            val_dataset = TensorDataset(val_seq, val_fm, val_dnabert, val_labels)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            model, optimizer, loss_fn, scheduler = build_improved_model(
                input_shape=(max_length, 7), fm_embedding_dim=640, dnabert_embedding_dim=768,
                pos_weight_value=pos_weight_value, device=device
            )

            EPOCHS = 50
            best_val_loss = float('inf')
            patience = 5
            patience_counter = 0

            for epoch in range(EPOCHS):
                model.train()
                train_loss = 0
                train_correct = 0
                for batch_seq, batch_fm, batch_dnabert, batch_y in train_loader:
                    optimizer.zero_grad()
                    logits = model(batch_seq, batch_fm, batch_dnabert)
                    loss = loss_fn(logits.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                    train_correct += (preds == batch_y).sum().item()

                model.eval()
                val_loss = 0
                val_correct = 0
                with torch.no_grad():
                    for batch_seq, batch_fm, batch_dnabert, batch_y in val_loader:
                        logits = model(batch_seq, batch_fm, batch_dnabert)
                        loss = loss_fn(logits.squeeze(), batch_y)
                        val_loss += loss.item()
                        preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                        val_correct += (preds == batch_y).sum().item()

                avg_val_loss = val_loss / len(val_loader)
                print(
                    f'Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss / len(train_loader):.4f}, '
                    f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_correct / len(val_dataset):.4f}'
                )
                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'./models/Improved_IRES_model_{fold_count}.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        break

            model.load_state_dict(torch.load(f'./models/Improved_IRES_model_{fold_count}.pt'))
            model.eval()

            test_dataset = TensorDataset(test_seq_data, test_fm_embeddings, test_dnabert_embeddings, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            test_pred = []
            with torch.no_grad():
                for batch_seq, batch_fm, batch_dnabert, _ in test_loader:
                    logits = model(batch_seq, batch_fm, batch_dnabert)
                    probs = torch.sigmoid(logits.squeeze())
                    test_pred.append(probs.cpu())
            test_pred = torch.cat(test_pred, dim=0)
            test_pred_all.append(test_pred)

            Sn, Sp, Acc, MCC = show_performance(test_labels.cpu(), test_pred)
            AUC = roc_auc_score(test_labels.cpu().numpy(), test_pred.numpy())
            print(f'Sn = {Sn:.4f}, Sp = {Sp:.4f}, Acc = {Acc:.4f}, MCC = {MCC:.4f}, AUC = {AUC:.4f}')

            all_Sn.append(Sn)
            all_Sp.append(Sp)
            all_Acc.append(Acc)
            all_MCC.append(MCC)
            all_AUC.append(AUC)

        test_pred_all = torch.stack(test_pred_all, dim=1).numpy()
        ensemble_pred = test_pred_all.mean(axis=1)
        sv_Sn, sv_Sp, sv_Acc, sv_MCC = show_performance(test_labels.cpu(), ensemble_pred)
        sv_AUC = roc_auc_score(test_labels.cpu().numpy(), ensemble_pred)
        sv_result = [sv_Sn, sv_Sp, sv_Acc, sv_MCC, sv_AUC]
        sv_10_result.append(sv_result)

        fpr, tpr, _ = roc_curve(test_labels.cpu().numpy(), ensemble_pred, pos_label=1)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        plt.plot(fpr, tpr, label=f'ROC Cycle {cycle + 1} (AUC={sv_AUC:.4f})')

    print('---------------------- Soft Voting Ensemble Results ----------------------')
    print(np.array(sv_10_result))
    performance_mean(np.array(sv_10_result))

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(np.array(sv_10_result)[:, 4])
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC={mean_auc:.4f})', lw=2, alpha=.8)
    plt.title('ROC Curve of Improved Model with RNA-FM and DNABERT_2')
    plt.legend(loc='lower right')
    plt.savefig('./images/Improved_ROC_Curve_with_RNAFM_DNABERT2.jpg', dpi=1200, bbox_inches='tight')
    plt.show()
