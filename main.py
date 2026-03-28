from preprocessing import run_preprocessing
def main():
    print("PROJEKT KI: GYM MEMBERS ANALYSIS")

    # Schritt 1: Preprocessing mit CLEAN, ENCODING, SPLIT und SCALING.
    X_train, X_test, y_train, y_test = run_preprocessing()
    print("\nFertig! Bereit fur Training")

    # TODO: train_and_evaluate(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()