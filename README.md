Project Title
IRES-Bert: Internal Ribosomal Entry Site Prediction Based on Dynamic Weighted Ensemble Learning

Project Introduction
A deep learning-based prediction model for Internal Ribosome Entry Sites (IRES). This model integrates one-hot + NCP (nucleotide chemical property) hand-coded sequences, RNA-FM pre-trained embeddings, and DNABERT-2 pre-trained embeddings. Feature extraction and classification are performed using CNN, BiLSTM, Multi-head Attention, and fully connected layers.
The model performs well on imbalanced datasets. The use of 10-fold cross-validation and soft voting ensemble further enhances the stability and generalization ability of the predictions.

Environment Requirements
·  python==3.9.18
·  pytorch-cuda==11.6
·  scikit-learn==1.3.0
·  numpy==1.26.2
·  pandas==1.5.2
·  matplotlib==3.6.2

Data Preparation
Data format: CSV files containing two columns — seq (sequence string) and label (0/1).
Pre-computed embeddings (must be prepared in advance)
RNA-FM embeddings: Save as rna-fm/sequence_embeddings_tr.npy (training set) and rna-fm/sequence_embeddings_t.npy (test set), shape (n_samples, 640).
DNABERT-2 embeddings: Save as dnabert2/embeddings_tr.npy and dnabert2/embeddings_t.npy, shape (n_samples, 768).

Usage
1.Adjust data paths and embedding paths to match your environment
2.Run the script(python code.py)
3.Outputs:Console prints performance metrics (Sn, Sp, Acc, MCC, AUC) for each fold and the ensemble.
