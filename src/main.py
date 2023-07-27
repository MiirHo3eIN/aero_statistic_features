import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import feature_classnew as fc 
import dataset_tr as ds
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import pickle






features_dir = "../features/"

def extract_features(x_input, ch):
    features = []
    FREQ_CUTS = [(0,5),(6,10),(12,20),(22,30),(35,40),(40,50)]
    for sample in np.arange(0, x_input.shape[0]):
        features_ = fc.features(FREQ_CUTS= FREQ_CUTS)
        feature_1, feature_1_name = features_.std_dev(x_input[sample, ch, :])
        feature_2 , feature_2_name = features_.skewness(x_input[sample, ch, :])
        feature_3 , feature_3_name = features_.kurtosis(x_input[sample, ch, :])
        feature_4 , feature_4_name = features_.mean(x_input[sample, ch, :])
        feature_5, feature_5_name = features_.ZCR(x_input[sample, ch, :])
        feature_6, feature_6_name = features_.RMSP(x_input[sample, ch, :])
        feature_7, feature_7_name = features_.EEPD((fs, x_input[sample, ch, :]))
        feature_8, feature_8_name = features_.CF((fs, x_input[sample, ch, :]))
        feature_9, feature_9_name = features_.spectral_features((fs, x_input[sample, ch, :]))
        feature_10 , feature_10_name = features_.MFCC((fs, x_input[sample, ch, :]))

        features_sample = [feature_1, feature_2, feature_3, feature_4, feature_5,feature_6] 
        features_sample.extend(feature_7)
        features_sample.extend([feature_8])
        features_sample.extend(feature_9)
        features_sample.extend(feature_10)
        #features_sample.extend([1,2])
        
        features.append(features_sample)
    return features

# Path: aerosense_scripts/main.py
FEATURE_EXTRACTION = True
EVALUATION = True
fs = 100 # Sampling frequency
n_estimators = 800
del_cells = [0, 23, 37]
cols = np.arange(0, 41)
use_cols_ = np.delete(cols, del_cells)
def main(train_experiments, test_experiments, folder_path):
    
    seq_len = 100 # 1 seconds of data
    train_x , train_y = ds.TimeSeriesDataset(folder_path, train_experiments, seq_len = seq_len)    
    test_x, test_y = ds.TimeSeriesDataset(folder_path, test_experiments, seq_len = seq_len) 
    #valid_x, valid_y = ds.TimeSeriesDataset(folder_path, valid_experiments, seq_len = seq_len) 
    

    train_x = train_x.detach().numpy()
    train_y = np.squeeze(train_y.detach().numpy()[:,0])
    test_x = test_x.detach().numpy()
    test_y = np.squeeze(test_y.detach().numpy()[:,0])

    if FEATURE_EXTRACTION: 
        
        for ch in use_cols_:
            print('--------------------------------------------------------------')
            print("Extracting features from training data for channel: ", ch)
            
            train_features = extract_features(train_x, ch)
            test_features = extract_features(test_x, ch)

            print('--------------------------------------------------------------')
            print("Save Training features")
            np.save(f"{features_dir}train_features_{ch}.npy", train_features)
            np.save(f"{features_dir}test_features_{ch}.npy", test_features)
            print('--------------------------------------------------------------')
            print(f"train_features shape: {np.array(train_features).shape}")
            print(f"test_features shape: {np.array(test_features).shape}")

            exit(0)


    if EVALUATION and not FEATURE_EXTRACTION: 
        
        for ch in use_cols_:

            print("Load Training features")
            train_features = np.load(f"{features_dir}train_features_{ch}.npy")
            test_features = np.load(f"{features_dir}test_features_{ch}.npy")

    # fit a random forest classifier 
            rfc = RandomForestClassifier(n_estimators,max_features="sqrt")
            rfc.fit(train_features, train_y)

            #predict test results
            y_rfc = rfc.predict(test_features)
            # print(y_rfc.shape)
            # print(test_y.shape)
            # check model accuracy score
            print('RFC model accuracy score with '+str(n_estimators)+' decision trees:{0:0.4f}'. format(accuracy_score(test_y , y_rfc )))
            cmatrix_rfc = confusion_matrix(test_y, y_rfc)

        # plt.figure()
        # with plt.style.context({'axes.labelsize':24,
        #                         'xtick.labelsize':14,
        #                         'ytick.labelsize':14}):
        #     ax = sns.heatmap(cmatrix_rfc/np.sum(cmatrix_rfc), annot=True, fmt='.1%',cmap='Blues', annot_kws={'size':14})
        # #plt.savefig('cmatrix_rfc8_800_sqrt.pdf')
        # plt.show()
        # #sns.heatmap(cmatrix_rfc/np.sum(cmatrix_rfc), annot=True, 
        #            fmt='.2%', cmap='Blues')
        #plt.show()

        #cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['healthy and 5 mm', '10mm and added mass', '15 and 20mm'])
        #cm_display.plot()
        #plt.show()


            print('--------------------------------------------------------------')
            print('Classification details random forest classifier with '+str(n_estimators)+' trees')
            print(f'Confusion matrix for channel {ch}\n\n\ {cmatrix_rfc}')

            print(classification_report(test_y,y_rfc))
            print('--------------------------------------------------------------')


if __name__ == "__main__":
    
    
    folder_path = "/home/miir_ho3ein/project/aerosense_CAD/cp_data/AoA_0deg_Cp"
    zeroing_experiments = [0, 1, 2, 6, 10, 11, 15, 19, 20, 25, 29, 30, 34, 38, 39,40, 44, 48,49, 53, 57, 58, 59, 63, 67,68, 72, 76, 77, 78, 82, 86, 87, 91, 95, 96, 97, 101, 105,106, 110] 
    test_experiments = [5,9, 24, 28, 43, 47, 62, 66, 81, 85, 100, 104]
    valid_experiments = [14, 18, 33, 37, 52, 56, 71, 75, 90, 94, 109, 113]
    train_experiments = np.delete(np.arange(0, 114), np.concatenate((test_experiments, valid_experiments, zeroing_experiments)))   

    print("Train experiments: ", train_experiments)
    print("Test experiments: ", test_experiments)
    print("Validation experiments: ", valid_experiments)
    
    
    main(train_experiments, test_experiments, folder_path)

