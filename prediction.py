import pandas as pd

def make_predictions(models, X_test):
    for name, model in models.items():
        preds = model.predict(X_test)

        df_preds = pd.DataFrame(preds, columns=[f"{name}_prediction"])
        df_preds.to_csv("outputs/predictions.csv", index=False)

        print(f"{name} predictions saved")