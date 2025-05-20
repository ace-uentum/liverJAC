# Survival Prediction in Liver Cirrhosis Patients Using LiverJAC Model – Deep Neural Networks

## 👥 Authors
| Name              | Role                      | Email                        |
|-------------------|---------------------------|------------------------------|
| Julius Rey Gida   |  Co-author / Contributor  | 24-01013@g.batstate-u.edu.ph |
| Christine Joy Lao |  Co-author / Contributor  | 24-00273@g.batstate-u.edu.ph |
| Allainer Reyes    |  Co-author / Contributor  | 24-01741@g.batstate-u.edu.ph |

- This repository contains the implementation of LIVERJAC, a deep neural network model design for predicting survival outcomes for patients with liver cirrhosis.
  It used Pytorch for training and evaluation.

## Methodology

**Requirements**
- Python 3.x
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib


### Data Preprocessing

- The dataset is loaded from a CSV file. `cirrhosis.csv`
- Columns with more than 20% missing data are dropped.
- Remaining missing values are removed.
- Categorical features are label-encoded.
- The target variable (`Target`) is binary: `0` for survived and `1` for died.
- Features are standardized using `StandardScaler` for improved neural network performance.
- The dataset is split into:
  - 72% training
  - 8% validation (further split from training)
  - 20% testing
This splits prevent data leakage and ensures a proper model evaluation



### Model Architecture: LiverJAC

The model is a deep neural network with three fully connected layers, each followed by:

- **Batch Normalization**: stabilizes and accelerates training by normalizing activations, reducing internal covariate shift.
- **ReLU Activation**: introduces non-linearity.
- **Dropout (0.3 rate)**: randomly disables neurons during training to prevent overfitting by encouraging redundancy.

Weights are initialized using **He initialization** for efficient training. *Kaiming He*

```python
import torch.nn as nn

class LiverJAC(nn.Module):
    def __init__(self, input_dim):
        super(LiverJAC, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(0.3)
        self.out = nn.Linear(16, 1)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.out(x)
        return x
```
### Training Process
- Loss Function: BCEWithLogitsLoss with class weighting to address class imbalance.
- Optimizer: AdamW, which combines Adam optimization with weight decay for regularization.
- Learning Rate Scheduler: Reduces learning rate on plateau to fine-tune learning.
- Early Stopping: Training stops if validation loss doesn’t improve after 10 epochs, avoiding overfitting.
- Batch Size: 32 samples per batch, shuffled for better generalization.
```python
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

```
### Evaluation and Threshold Selection
- Predictions are probabilistic outputs (sigmoid applied).
- Precision-Recall Curve used to identify the best threshold maximizing F1-score.
- Additionally, a threshold is chosen to ensure at least 80% recall, prioritizing sensitivity (detecting as many deaths as possible).
- Final evaluation includes accuracy, AUC, confusion matrix, and detailed classification reports.

```python
from sklearn.metrics import precision_recall_curve, classification_report

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

desired_recall = 0.8
valid_indices = np.where(recall[:-1] >= desired_recall)[0]
if len(valid_indices) > 0:
    valid_f1 = f1_scores[valid_indices]
    best_idx = valid_indices[np.argmax(valid_f1)]
    threshold_for_recall = thresholds[best_idx]
else:
    threshold_for_recall = best_threshold

print(classification_report(y_test, (y_pred_prob >= threshold_for_recall).astype(int)))
```
**IMPORTANT NOTES**
- Batch Normalization helps reduce internal covariate shift, making training faster and more stable.
- Choosing 80% recall prioritizes correctly identifying patients who will not survive, important in medical contexts where missing a positive case has high cost.
- Dropout mitigates overfitting by randomly disabling neurons during training, which encourages the network to learn robust patterns rather than noise.

### Experiments & Fine-Tuning Strategies
Here are five fine-tuning strategies our group has explored to test their impact on performance. The original model settings already performed well, so these experiments were designed to explore trade-offs and potential improvements.

**Original Setup (Best Performance)**
**Learning Rate: 0.001**
**Dropout Rate: 0.3**
**Batch Size: 32**
**Early Stopping Patience: 10**
**Class Imbalance Weight: pos_weight = 1.5**

### Slowing Down the Learning Rate
```python
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
```
Learning rate reduced from 0.001 → 0.0005. A slower learning rate might allow the model to converge more smoothly. 
However, it sometimes made training slower or less decisive, and didn’t outperform the original.

### Tweaking Dropout to Keep More Neurons Active
```python
self.dropout1 = nn.Dropout(0.2)
self.dropout2 = nn.Dropout(0.2)
self.dropout3 = nn.Dropout(0.2)
```
Dropout reduced from 0.3 → 0.2.Lower dropout means more neurons stay active. This helped the model train faster but slightly increased overfitting risk. 
The original 0.3 gave better generalization.

### Using Bigger Batches
```python
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
```
The batch size increased from 32 to 64. Bigger batches train faster per epoch, but sometimes reduce model's ability to generalize. 
Original smaller batches worked better for this dataset.

### Giving the Model More Patience

patience = 15  - here Original was 10
- Early stopping patience increased from 10 to 15. Let the model try harder before stopping. But longer training sometimes led to overfitting. 
The original patience of 10 hit a better result.

### Adjusting for Class Imbalance
```python
pos_weight = torch.tensor([2.0])  # Original was 1.5
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```
The pos_weight increased from 1.5 to 2.0. Deaths were underrepresented, so boosting the weight forces the model to pay more attention to them. 
However, the original 1.5 balance gave better precision-recall results.

### Summary & Reflections
After testing several fine-tuning strategies, we found that the original configuration—with a moderate learning rate, balanced dropout, reasonable batch size, and early stopping—consistently yielded the best results in terms of accuracy, AUC, and generalization.

**Here’s what stood out:**
- He initialization, batch normalization, and dropout (at 0.3) worked together to stabilize training and prevent overfitting.
- Balanced loss weighting (pos_weight = 1.5) was crucial to handle the class imbalance without skewing the predictions.
- Patience of 10 gave the model just enough time to learn while avoiding unnecessary training.

### Future directions:
- Try more advanced techniques like focal loss or SMOTE for better class imbalance handling.
- Explore ensemble methods (combining predictions from multiple models).
- Use automated hyperparameter tuning libraries like Optuna or Ray Tune for even better parameter optimization.


