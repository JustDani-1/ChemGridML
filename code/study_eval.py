import optuna
import optuna.visualization as vis
import plotly.io as pio

# Set the renderer to browser (opens in default browser)
pio.renderers.default = "browser"

study = optuna.load_study(
    study_name="ECFP_XGBoost_BBB_Martins",
    storage="sqlite:///./studies/1756218199/studies.db"
)

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")

vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()

print("test")
