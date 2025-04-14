
from src.datapreprocessing import preprocess_data
from src.model import train_models
from src.evaluate import evaluate_model
from src.eda import perform_eda
from src.data_loader import load_data

# Step 1: Load Data
df = load_data('data/Churn_Modelling.csv')

# Step 2: EDA
perform_eda(df)

# Step 3: Preprocess
X_train, X_test, y_train, y_test = preprocess_data()

# Step 4: Train Models
models = train_models(X_train, y_train)

# Step 5: Evaluate
print("\n Evaluating all models:\n")
for name, model in models.items():
    print(f"\n {name} Results:")
    evaluate_model(model, X_test, y_test)
