from sklearn.metrics import r2_score


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
):

    report = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        score = r2_score(
            y_test,
            prediction
        )

        report[name] = score

    return report