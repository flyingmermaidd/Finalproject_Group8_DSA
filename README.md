
# Loan Approval Prediction using Machine Learning

This project presents a comprehensive solution for predicting loan approval based on applicant data using machine learning techniques. The workflow includes data preprocessing, exploratory data analysis (EDA), and the development of a predictive model using Random Forest Classification. The goal of this project is to predict whether a loan application will be approved based on various applicant and loan-related features.

## Project Structure

```
├── Business_Requirement_Document.pdf
├── Technical_Design_Document.pdf
├── problemset3.ipynb
├── datasets/
│   ├── train.csv
│   ├── test.csv
├── outputs/
│   ├── updated_train.csv
│   ├── updated_test.csv
├── assets/
│   ├── loan-prediction.png
├── README.md
```

## Objective

The goal of this project is to develop a machine learning model that predicts loan approval based on applicant features such as income, loan amount, credit history, and other related factors. By leveraging data science techniques, the project aims to create an efficient, automated decision-making system for financial institutions.

## Documentation

- **Business Requirement Document (BRD)**: Describes the business context, problem statement, and success criteria for the loan approval prediction system.
- **Technical Design Document (TDD)**: Provides a detailed explanation of the system architecture, data pipeline, feature engineering, model selection, and performance evaluation.

## Dataset Overview

The dataset used in this project contains information about loan applicants and their loan statuses. Key features include:

- **Loan_ID**: Unique identifier for each loan application
- **Gender, Married, Dependents**: Demographic information
- **ApplicantIncome, CoapplicantIncome**: Income-related data
- **LoanAmount**: The loan amount requested
- **Credit_History**: Whether the applicant has a credit history
- **Loan_Status**: Target variable (Approved/Not Approved)

Initially, the dataset contained 614 records, with columns for demographic and loan-related information. After preprocessing, the dataset was reduced to 418 rows for training and 263 rows for testing.

## Data Preprocessing

The following steps were taken to clean and prepare the data for modeling:

1. **Removal of Duplicates and Null Values**: All duplicate and missing records were handled to ensure data integrity.
2. **Categorical Encoding**: Categorical variables like `Gender`, `Married`, and `Education` were converted into numerical formats using one-hot encoding.
3. **Outlier Handling**: Outliers were detected and removed using both the Z-score and IQR methods to ensure the data quality.
4. **Normalization**: Numerical features were normalized to standardize the scale for machine learning algorithms.

## Exploratory Data Analysis (EDA)

EDA was conducted to understand the relationships between features and their impact on loan approval. Key findings include:

- **Loan Amount vs. Applicant Income**: A scatter plot revealed a correlation between income levels and the requested loan amount.
- **Credit History vs. Loan Status**: A count plot highlighted the influence of a clean credit history on loan approval.
- **Property Area vs. Loan Status**: Regional differences in loan approval were observed, with urban areas seeing higher approval rates.

These insights informed the feature engineering and model selection process.

## Model Development

### Model Selection

The project used the **Random Forest Classifier**, a robust ensemble method known for its performance and interpretability. The dataset was split into training (80%) and testing (20%) subsets. The following steps were taken:

1. **Data Splitting**: The independent variables (features) and dependent variable (target: `Loan_Status`) were separated.
2. **Model Training**: The Random Forest classifier was trained on the preprocessed training data.
3. **Model Evaluation**: The model was evaluated using accuracy, recall (True Positive Rate), and error rate.

### Results

- **Accuracy**: 74%
- **Recall (True Positive Rate)**: 89%
- **Error Rate**: 26%
- **Confusion Matrix**: Perfect classification on the test set with no false positives or false negatives.

Despite the small size of the dataset, the model performed well, indicating that the data had strong predictive value for loan approval. However, some issues were noted regarding the illogical loan distribution to applicants with low income, which might affect real-world application.

## Conclusion

This project demonstrated the use of machine learning to solve the loan approval prediction problem. The Random Forest model was able to classify loan approval status with reasonable accuracy and recall. However, further improvements can be made by addressing data imbalance, tuning hyperparameters, and incorporating more sophisticated models.

## Future Improvements

- **Feature Engineering**: Experiment with new features or derived variables to improve model performance.
- **Model Optimization**: Apply techniques like hyperparameter tuning and cross-validation to enhance model accuracy.
- **Handling Data Imbalance**: Use methods such as SMOTE (Synthetic Minority Over-sampling Technique) to address any class imbalance.
- **Alternative Models**: Implement other classification algorithms like XGBoost or LightGBM to compare performance.

## References

- [Original Kaggle Dataset & Tutorial](https://www.kaggle.com/code/talhabu/loan-eligibility-tutorial-from-scratch-to-advance)

## Acknowledgments

Thank you for reviewing this project. Should you have any feedback or inquiries, feel free to reach out for collaboration or further discussion.
