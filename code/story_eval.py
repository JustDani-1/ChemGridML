import joblib
import optuna.visualization as vis

# Load the study
study = joblib.load('optuna_study_03.pkl')

# Now you have everything back!
print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")

# Create all visualizations
vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()
# etc.