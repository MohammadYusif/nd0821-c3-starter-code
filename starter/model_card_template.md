# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a supervised binary classification model trained to predict whether an individual’s income exceeds $50K per year. It is implemented using Logistic Regression from scikit-learn.

*   **Model type:** Logistic Regression
*   **Framework:** scikit-learn
*   **Input:** Tabular demographic and employment features
*   **Output:** Binary classification (<=50K or >50K)
*   **Artifacts saved:** Trained model, categorical encoder, label binarizer

## Intended Use

The model is intended for educational and demonstrational purposes only. It demonstrates how to:

*   Train a machine learning model on tabular census data
*   Evaluate performance using classification metrics
*   Deploy a trained model behind a REST API using FastAPI
*   Perform slice-based performance analysis

The model should not be used for real-world decision-making related to employment, finance, credit scoring, or social services.

## Training Data

The training data is derived from the U.S. Census Income dataset, which includes demographic and employment-related attributes such as:

*   Age
*   Education
*   Workclass
*   Occupation
*   Marital status
*   Hours worked per week
*   Capital gains/losses
*   Race and sex

The dataset was cleaned by stripping unnecessary whitespace and encoding categorical variables using one-hot encoding. The target label is `salary`, indicating whether income is <=50K or >50K.

## Evaluation Data

The dataset was split into training and test sets using an 80/20 split. The test set was held out during training and used exclusively for evaluating model performance.

In addition to overall evaluation, the model was evaluated on data slices based on categorical features (e.g., education level). Slice performance metrics were written to `slice_output.txt`.

## Metrics

The following metrics were used to evaluate model performance:

*   **Precision**
*   **Recall**
*   **F1-score** (F-beta with β = 1)

On the held-out test set, the model achieved approximately:

*   **Precision:** ~0.75
*   **Recall:** ~0.60
*   **F1-score:** ~0.67

*(Exact values may vary slightly depending on the train/test split.)*

Slice-based metrics were computed for different values of categorical features to identify performance differences across subgroups.

## Ethical Considerations

This model is trained on historical census data, which may reflect existing social and economic biases. Predictions may be influenced by sensitive attributes such as race, sex, or marital status.

Using this model in real-world decision-making could result in unfair or discriminatory outcomes. Care must be taken to avoid reinforcing historical inequalities.

## Caveats and Recommendations

*   The model is relatively simple and does not capture complex nonlinear relationships.
*   Performance varies across different data slices, indicating potential bias or underperformance for certain groups.
*   The dataset may be outdated and not representative of current socioeconomic conditions.
*   The model should not be deployed in production systems without extensive bias analysis, validation, and monitoring.

Future improvements could include:

*   Using more advanced models
*   Performing fairness-aware training
*   Regularly updating the dataset
*   Conducting deeper bias and error analysis
