import pandas as pd
import pickle

def predict(X_test):
    # load the model and the pipeline
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)

    predictions = pd.DataFrame(model.predict(pipeline.transform(X_test)), columns=['LeaveOrNot'],
                 index=X_test.index)
    return predictions

def main():
    # read X_test
    X_test = pd.read_csv('X_test.csv', index_col=0)
    predictions = predict(X_test)
    predictions['LeaveOrNot'].to_json('y_test_pred.json')


if __name__ == '__main__':
    main()
