# Credit-card-default-prediction-with-SVM
This project aims to build and optimize a classification model based on Support Vector Machines (SVM) to predict bank customer defaults. The study focuses on handling real-world data challenges such as class imbalance and high feature multicollinearity.
The project was developed as part of the Artificial Intelligence Methods laboratory course.

# Key Workflow**
Exploratory Data Analysis (EDA): Identified key patterns and strong correlations between financial features.

**Feature Engineering:**
- Addressed multicollinearity by aggregating highly correlated billing features (BILL_AMT1-6) into a single AVG_BILL metric.
- Applied data normalization using StandardScaler to ensure optimal SVM performance.

**Model Optimization:**
- Performed hyperparameter tuning for the RBF kernel (C and gamma) using GridSearchCV.
- Analyzed the impact of different parameter values on model generalization and decision boundaries.

**Handling Class Imbalance:**
- Implemented class_weight='balanced' to prioritize the minority class (defaulters).
- Shifted focus from standard Accuracy to more robust metrics like Balanced Accuracy and F1-score.

**Results & Performance**
The final model demonstrates a high capability for generalization, with consistent results between training and validation sets:
- Balanced Accuracy: ~69.2% (Validation)
- F1-score (Class 1): 0.51
- Recall (Sensitivity): 54% — the model successfully identifies over half of the actual default cases.

**Business Interpretation**
In credit risk assessment, the cost of missing a "default" (False Negative) is significantly higher than the cost of a "false alarm" (False Positive). This model achieves a strategic balance: by accepting a 49% precision rate, the bank can proactively secure over 50% of potentially lost credit capital.

# Tech stack: 
- Language: Python
- Libraries: Pandas, NumPy, Matplotlib, Seaborn
- Machine Learning: Scikit-learn (SVM, Model Selection, Preprocessing)

  
