# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Model](https://img.shields.io/badge/Model-Random%20Forest-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-yellow)

A machine learning-based approach to predict **customer churn** for a subscription-based service. This project utilizes historical customer data to predict whether a customer is likely to leave the service, helping businesses take proactive actions.

## Objective

To build an intelligent model that can accurately **predict customer churn**, enabling businesses to identify at-risk customers and implement retention strategies to reduce churn rates.

## Key Deliverables

-  **Exploratory Data Analysis** (EDA)
-  **Data preprocessing** and **feature engineering**
-  **Random Forest** model for churn prediction
-  Model evaluation with key metrics (Accuracy, Precision, Recall, F1)
-  Model export as `.pkl` for reuse
-  Custom script (`predict.py`) for predictions on new data
-  Dataset managed using **Git LFS**

## Technologies & Tools Used

| Category           | Tools / Libraries                               |
|--------------------|--------------------------------------------------|
| Language           | Python 3.8+                                      |
| Libraries          | pandas, numpy, seaborn, matplotlib, scikit-learn |
| ML Model           | Random Forest                                    |
| Version Control    | Git + GitHub                                     |
| Large File Storage | Git LFS                                          |

## Project Structure

Customer Churn Prediction/ ├── data/ │ └── Churn_Modelling.csv # Dataset file ├── src/ │ ├── init.py │ ├── datapreprocessing.py # Data loading and preprocessing code │ ├── eda.py # Exploratory Data Analysis (EDA) code │ ├── models.py # Model training code │ ├── evaluation.py # Model evaluation code ├── outputs/ │ └── churn_distribution.png # Plot generated during EDA ├── churn_prediction.py # Main script to run the project └── README.md # Project documentation



## Model Performance

| Metric       | Score    |
|--------------|----------|
| Accuracy     | 86.75%   |
| Precision    | 0.88     |
| Recall       | 0.96     |
| F1-Score     | 0.92     |

> **Note: These scores are based on the Random Forest model trained on the dataset.**

### Confusion Matrix  

![Confusion Matrix](https://github.com/user-attachments/assets/4f0efeb8-5244-4b75-9fcf-3e5c0b9c035e)

### Feature Importance  

![Feature Importance](https://github.com/user-attachments/assets/5fe25dbf-039c-4800-87a3-d1cc436c2d0c)

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Codsoft.git
    cd Codsoft/Customer Churn Prediction
    ```

2. Run the project:
    ```bash
    python churn_prediction.py
    ```

This will execute the **churn_prediction.py** script which performs the following:
- Loads and preprocesses the dataset.
- Performs exploratory data analysis (EDA).
- Trains machine learning models (Random Forest).
- Evaluates and visualizes model performance.

## Conclusion

This project successfully demonstrates the prediction of customer churn using machine learning algorithms. The **Random Forest** model offers an excellent balance between accuracy and performance metrics, helping to identify at-risk customers effectively.

### Next Steps:
- Implement hyperparameter tuning for model optimization.
- Deploy the model to a production environment.
- Explore feature engineering to further improve model accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Dataset: **Bank Customer Churn Prediction** from Kaggle.
- Machine Learning Libraries: **scikit-learn**, **pandas**, **numpy**, **matplotlib**, and **seaborn**.

