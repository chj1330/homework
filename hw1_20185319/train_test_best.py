# GCT634 (2018) HW1
#
# Mar-18-2018: initial version
#
# Juhan Nam
#


from feature_summary import *
from sklearn.externals import joblib


if __name__ == '__main__':

    # load data 
    train_X = mean_mfcc('train')
    valid_X = mean_mfcc('valid')
    test_X = mean_mfcc('test')

    # label generation
    cls = np.array([1,2,3,4,5,6,7,8,9,10])
    train_Y = np.repeat(cls, 100)
    valid_Y = np.repeat(cls, 20)
    test_Y = np.repeat(cls, 20)

    # feature normalizaiton
    train_X = train_X.T
    train_X_mean = np.mean(train_X, axis=0)
    train_X = train_X - train_X_mean
    train_X_std = np.std(train_X, axis=0)
    train_X = train_X / (train_X_std + 1e-5)
    
    valid_X = valid_X.T
    valid_X = valid_X - train_X_mean
    valid_X = valid_X/(train_X_std + 1e-5)

    final_model = joblib.load('final_model_87.0.pkl')
    print(final_model)
    # now, evaluate the model with the test set
    test_X = test_X.T
    test_X = test_X - train_X_mean
    test_X = test_X/(train_X_std + 1e-5)
    test_Y_hat = final_model.predict(test_X)

    accuracy = np.sum((test_Y_hat == test_Y))/200.0*100.0
    print 'test accuracy = ' + str(accuracy) + ' %'

    valid_Y_hat = final_model.predict(valid_X)

    accuracy = np.sum((valid_Y_hat == valid_Y))/200.0*100.0
    print 'valid accuracy = ' + str(accuracy) + ' %'

    train_Y_hat = final_model.predict(train_X)
    accuracy = np.sum((train_Y_hat == train_Y))/1000.0*100.0
    print 'train accuracy = ' + str(accuracy) + ' %'

