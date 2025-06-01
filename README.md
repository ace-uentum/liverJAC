# Application of LiverJAC-MHA with Gating for Predicting Outcomes in Liver Cirrhosis

## 👥 Authors
| Name              | Role                      | Email                        |
|-------------------|---------------------------|------------------------------|
| Julius Rey Gida   |  Co-author / Contributor  | 24-01013@g.batstate-u.edu.ph |
| Christine Joy Lao |  Co-author / Contributor  | 24-00273@g.batstate-u.edu.ph |
| Allainer Reyes    |  Co-author / Contributor  | 24-01741@g.batstate-u.edu.ph |

- This repository presents **LiverJAC-MHA**, a deep learning framework developed for graduate-level research in the field of medical machine learning. The model is designed to predict patient survival outcomes in liver cirrhosis using structured clinical data. LiverJAC-MHA incorporates modern deep learning components to capture complex interactions among clinical variables and enhance predictive accuracy.

---
## Abstract
Liver cirrhosis presents a significant burden to global health, reinforcing the need for accurate survival prediction
tools that are reliable and data-driven to help with informed clinical decisions. This study demonstrates LiverJAC-MHA
with Gating, a deep neural network model trained using clinical and laboratory information from the cirrhosis.csv
dataset. The model features multi-head self-attention to look at prior sequences of features in addition to using dynamic
feature gating to improve interpretability and performance.The model was trained following rigorous data
preprocessing with the use of five-fold cross validation for validation. The performance of the deep learning model indicated an average accuracy of 81% with an AUC of 0.84.
The new model outperformed traditional methods including the MELD and Child-Pugh scores. All three of the most influential features linked to predicting survival were
significant clinical measures (bilirubin, albumin, and prothrombin time). The study provides evidence that deep learning models such as LiverJAC-MHA can provide more
accurate and interpretable survival predictors for cirrhosis, and thus have a lot of potential towards helping clinical pathways to provide individualized care for patients and
earlier identification of high-risk patients in cirrhosis management.
## Key Features
- **Multi-Head Self-Attention (MHA):** Captures complex interactions between clinical features.
- **Residual Blocks & Layer Normalization:** Supports deep model stability and faster convergence.
- **Feature Gating:** Learns feature-level importance during training.
- **Cross-Validation:** 5-fold stratified evaluation for robust performance estimation.
- **Performance Metrics:** Includes ROC-AUC, F1-optimized thresholding, confusion matrix, and bootstrapped confidence intervals.

## Dataset Information
The dataset (`cirrhosis.csv`) is a cleaned version of a publicly available clinical dataset on liver cirrhosis, containing:
- **Demographics:** Age, Sex
- **Laboratory Tests:** Bilirubin, Albumin, INR, etc.
- **Outcome Variable:** `Status` (0 = Survived, 1 = Died)

**Preprocessing Steps:**
- Features with >20% missing values removed
- Complete-case filtering
- Min-Max normalization
---
## Tools:
- Python with PyTorch and Sklearn
- Developed and tested in Google Colab
---
## Code Overview
### Model Architecture with Multi-Head Attention
```python  
class LiverJAC_MHA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, dropout):
        super(LiverJAC_MHA, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(0)  # For multi-head attention input shape (L, N, E)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.squeeze(0)
        out = self.dropout(attn_output)
        out = self.fc2(out)
        return self.sigmoid(out)
``` 
### Dynamic Gating
```python
class DynamicFeatureGating(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Linear(input_dim, input_dim)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
    def forward(self, x):
        gate = torch.sigmoid(self.gate(x))
        return x * gate
```
### Data Splitting with Stratified 5-Fold Cross-Validation
Model initialization, training, and validation steps...
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
```
### Bootstrapped Confidence Interval Calculation for Metrics
```python
def bootstrap_ci(metric_fn, y_true, y_pred, n_bootstraps=1000, alpha=0.95):
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.sort(bootstrapped_scores)
    lower = np.percentile(sorted_scores, ((1 - alpha) / 2) * 100)
    upper = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)
    return lower, upper

accuracy_ci = bootstrap_ci(accuracy_score, y_true, y_pred)
```

### Confusion Matrix Plotting
```python
def plot_confusion_matrix(cm, classes):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('Confusion matrix')
    plt.xticks(np.arange(len(classes)), classes)
    plt.yticks(np.arange(len(classes)), classes)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

cm = confusion_matrix(y_val, y_pred_bin)
plot_confusion_matrix(cm, classes=['Survived', 'Died'])
```
## Models Implemented
- **LiverJAC-MHA** vs tradional scoring method : MELD & Child-pugh
- **Logistic Regression**
- **Decision Tree**
---
## Results and Discussion
To evaluate the prognostic performance of our proposed model, LiverJAC-MHA (a Deep Neural Network enhanced with Multi-Head Self-Attention and clinical feature integration), we conducted a 5-fold cross-validation on a preprocessed cirrhosis patient dataset. The model’s classification outcomes were benchmarked against traditional clinical scoring systems — MELD and Child-Pugh — as well as baseline machine learning classifiers, including Logistic Regression and Decision Tree models.
The evaluation metrics included Accuracy, Area Under the Receiver Operating Characteristic Curve (AUC), Precision, Recall (Sensitivity), and F1-score. These metrics were selected to comprehensively assess both overall model performance and its sensitivity to the minority class, which in clinical prognosis is critical for identifying at-risk patients.
The table below summarizes the performance comparison:
| Model               | Accuracy | AUC  | Precision | Recall (Sens) | F1   |
|---------------------|---------|------|-----------|--------------|------|
| JAC-MHA            | 0.807   | 0.83 | 0.767     | 0.767        | 0.754 |
| MELD               | 0.707   | 0.787 | 0.586     | 0.793        | 0.672 |
| Child-Pugh         | 0.699   | 0.749 | 0.584     | 0.76         | 0.658 |
| Logistic Regression | 0.79   | 0.833 | 0.818     | 0.587        | 0.675 |
| Decision Tree      | 0.747   | 0.731 | 0.661     | 0.667        | 0.661 |

LiverJAC-MHA achieved the highest F1-score (0.754) and strong AUC (0.830), indicating a balanced ability to detect both survival and mortality outcomes.
While Logistic Regression recorded a slightly higher AUC (0.833), it suffered from lower recall (0.587), potentially missing a significant number of at-risk patients — a critical issue in clinical decision-making.
The MELD and Child-Pugh scores, although standard in hepatology, underperformed in terms of precision and F1-score, highlighting the limitations of static, rule-based approaches compared to data-driven deep learning methods.
LiverJAC-MHA's self-attention mechanism and MELD score integration likely contributed to its improved recall and F1, especially in handling nuanced interactions among features.
These results underscore the potential of incorporating advanced deep learning architectures with clinically relevant features for more accurate and actionable prognosis in liver cirrhosis patients. Future work may explore external validation on independent cohorts and integration with real-time electronic health records.

## Citations
E. Dickson, P. Grambsch, T. Fleming, L. Fisher, and A. Langworthy. "Cirrhosis Patient Survival Prediction," UCI Machine Learning Repository, 1989. [Online]. Available: https://doi.org/10.24432/C5R02G.

## Academic Context
- Course: Machine Learning (Master of Science in Information Technology)
- Institution: Batangas State University
- Professor: Dr. Montalbo
- Term: 2nd Semester, Academic Year 2024–2025

## License
This project is licensed under the MIT License. See the LICENSE file for details.



