from utils import import_data, data_wrangling, features, fit, print_stats, create_submission_file

if __name__ == "__main__":
    train_data, test_data = import_data()
    train_data, test_data = data_wrangling(train_data, test_data)
    X, y, X_test = features(train_data, test_data)
    model = fit(X, y)
    print_stats(X,y,model) #without crossvalidation
    create_submission_file(model, X_test, test_data)