from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import json

def evaluate_models(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        preds = model.predict(X_test)

        if name == "regression":
            results[name] = {
                "MSE": mean_squared_error(y_test, preds),
                "R2": r2_score(y_test, preds)
            }
        else:
            results[name] = {
                "Accuracy": accuracy_score(y_test, preds)
            }

    print("Evaluation Results:", results)

    with open("outputs/metrics.json", "w") as f:
        json.dump(results, f, indent=4)