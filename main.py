"""
Working script


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
import random

from train_SAE import RunModel
from preprocessing import Process_Train_Data, Process_Test_Data

from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, confusion_matrix, recall_score


# Labelling Excels rows:
excel_list = ['Average AUC', 'sd', 'Average Accuracy', 'sd', 'Average Precision', 'sd', 'Average Recall', 'sd']
for excel_iteration in range(10):  # Selecting how many runs to do to print in Excel

    def make_anomaly_list():
        """
        Creating binary list to show times of when failure happens.
        :return:
        """
        anomaly_list = [0] * len(timepoints)
        for i in range(0, len(timepoints)):
            # If any failure timepoint matches a timepoint, flag anomaly (1).
            if any(round((failure_timepoints - original_timestamps['%time'][0]) / 1000000000)
                   ==
                   round(timepoints['%time'][i])):
                anomaly_list[i] = 1
        return anomaly_list


    def mean_absolute_dev(data):
        """
        Returns deviation comparing it to the mean of the datapoints.
        :param data: array of data
        :return:
        """
        mean = np.mean(data)
        sums = 0
        for i in range(len(data)):
            dev = abs(data[i] - mean)
            sums = sums + round(dev, 2)
        return sums / (len(data) + 1)


    def find_outlier_cause(reconstructed, test):
        """
        Shows loss graph that compares test data with autoencoder generated data.
        :param reconstructed:
        :param test:
        :return:
        """
        p_losses = pd.DataFrame(abs(x_test_tensor - reconstructed) / x_test_tensor)
        fig, ax = plt.subplots()
        for i in range(0, 5):
            ax.scatter(x=p_losses.index[len(p_losses) - 200:len(p_losses)],
                       y=p_losses[i][len(p_losses) - 200:len(p_losses)], s=1, label=p_losses.columns[i])
        ax.legend()
        plt.show()

        fig, ax = plt.subplots()
        for i in range(5, 10):
            ax.scatter(x=p_losses.index[len(p_losses) - 200:len(p_losses)],
                       y=p_losses[i][len(p_losses) - 200:len(p_losses)], s=1, label=p_losses.columns[i])
        ax.legend()
        plt.show()

        fig, ax = plt.subplots()
        for i in range(10, 15):
            ax.scatter(x=p_losses.index[len(p_losses) - 200:len(p_losses)],
                       y=p_losses[i][len(p_losses) - 200:len(p_losses)], s=1, label=p_losses.columns[i])
        ax.legend()
        plt.show()

        fig, ax = plt.subplots()
        for i in range(15, 18):
            ax.scatter(x=p_losses.index[len(p_losses) - 200:len(p_losses)],
                       y=p_losses[i][len(p_losses) - 200:len(p_losses)], s=1, label=p_losses.columns[i])
        plt.show()


    # Manually selecting files related to engine failure for x_test data:
    test_flight_id01 = "carbonZ_2018-07-18-15-53-31_2_engine_failure"
    test_flight_id02 = "carbonZ_2018-07-18-16-22-01_engine_failure_with_emr_traj"
    test_flight_id03 = "carbonZ_2018-07-18-16-37-39_2_engine_failure_with_emr_traj"
    test_flight_id04 = "carbonZ_2018-07-30-16-29-45_engine_failure_with_emr_traj"
    test_flight_id05 = "carbonZ_2018-07-30-16-39-00_1_engine_failure"
    test_flight_id06 = "carbonZ_2018-07-30-17-10-45_engine_failure_with_emr_traj"
    test_flight_id07 = "carbonZ_2018-07-30-17-20-01_engine_failure_with_emr_traj"
    test_flight_id08 = "carbonZ_2018-07-30-17-36-35_engine_failure_with_emr_traj"
    test_flight_id09 = "carbonZ_2018-07-30-17-46-31_engine_failure_with_emr_traj"
    test_flight_id10 = "carbonZ_2018-09-11-14-22-07_2_engine_failure"
    test_flight_id11 = "carbonZ_2018-10-05-15-55-10_engine_failure_with_emr_traj"
    test_flight_id12 = "carbonZ_2018-10-18-11-04-35_engine_failure_with_emr_traj"

    test_flight_id22 = "carbonZ_2018-10-05-15-52-12_3_engine_failure_with_emr_traj"

    fids = [
                (test_flight_id01, "engines"),
                (test_flight_id02, "engines"),
                (test_flight_id03, "engines"),
                (test_flight_id04, "engines"),
                (test_flight_id05, "engines"),
                (test_flight_id06, "engines"),
                (test_flight_id07, "engines"),
                (test_flight_id08, "engines"),
                (test_flight_id09, "engines"),
                (test_flight_id10, "engines"),
                (test_flight_id11, "engines"),
                (test_flight_id12, "engines"),
                (test_flight_id22, "engines"),
              ]

    # Processing train data through preprocessing.py
    train = Process_Train_Data()

    # Create x and y train
    train_times = train['%time']
    x_train = train.drop('%time', inplace=True, axis=1)

    # Train stacked autoencoder
    autoencoder_model, x_train_tensor, scaler = RunModel(learning_rate=1e-3, squeeze=16, x_train=x_train)

    # Reconstruct train data and calculate final mse loss
    reconstructed = autoencoder_model.to("cpu").reconstruct(autoencoder_model.encode(x_train_tensor.float()))
    train_loss = tf.keras.losses.mse(x_train_tensor.float().detach().cpu().numpy(),
                                     reconstructed.float().detach().cpu().numpy())

    x_train = x_train.reset_index(drop=True)

    # Plotting train data with threshold
    plt.scatter(x=x_train.index, y=train_loss, s=1)
    train_threshold = np.mean(train_loss.numpy()) + (np.std(train_loss.numpy()))
    plt.axhline(train_threshold, color="red")
    plt.title(label="Training Data")
    plt.show()

    count = 0
    for i in range(0, len(train_loss)):
        if train_loss[i] > train_threshold:
            count += 1
    print(1 - (count/len(train_loss)))

    engine_acc_log = []
    engine_prec_log = []
    engine_auc_log = []
    engine_recall_log = []
    engine_sample_dp = pd.DataFrame()
    engine_full_dp = pd.DataFrame()

    # Iterate through test flight ids, calculate performance metrics + plot
    for fid in fids:
        # Process x_test data
        test = Process_Test_Data(fid[0])

        # Separate x and y test
        original_timestamps = test['%time']
        x_test = test.drop('%time', inplace=True, axis=1)
        original_timestamps = original_timestamps.reset_index()

        print(".............")
        print(fid[0])
        print("Train shape: " + str(x_train.shape))
        print("Test shape: " + str(x_test.shape))

        # Normalise data and convert to tensor
        test_dataset = pd.DataFrame.to_numpy(x_test)

        test_dataset = scaler.transform(test_dataset)
        x_test_tensor = torch.tensor(test_dataset.astype(float))

        # Inference stage and  calculate loss
        reconstructed = autoencoder_model.reconstruct(autoencoder_model.encode(x_test_tensor.float()))
        test_loss = tf.keras.losses.mse(reconstructed.float().detach().numpy(), x_test_tensor.float().detach().numpy())

        # Calculate anomaly threshold as mean and std deviation of loss, transform timestamps
        threshold = list()
        test_loss = test_loss.numpy()

        for i in range(0, len(test_loss)):
            threshold.append(np.mean(test_loss[0:i]) + mean_absolute_dev(test_loss[0:i]))
        # threshold =  np.mean(test_loss.numpy())  + (np.std(test_loss.numpy()))

        # Timepoints is the test recorded time
        timepoints = (original_timestamps - original_timestamps['%time'][0]) / 1000000000

        # Selecting a value for static threshold
        static_threshold = 1.5

        # Load true failure diagnostics
        failure_timepoints = pd.read_csv("processed\\" + fid[0] + "\\" + fid[0] + "-failure_status-" + fid[1] + ".csv")['%time']

        # Binary list for true labels
        anom_scores = make_anomaly_list()

        # Print feature losses graphs
        # find_outlier_cause(reconstructed,test_dataset)

        # Create binary list for label predictions
        col_names = x_test.columns
        outlier_causes = []
        anom_pred_scores = [0] * len(timepoints)
        for x in range(0, len(test_dataset)):
            col_names = x_test.columns
            if test_loss[x] > threshold[x]:  # add +1 anomaly when loss > threshold
                anom_pred_scores[x] = 1
            else:
                anom_pred_scores[x] = 0

        safe_only = list(filter(lambda x:  anom_scores[x] == 0, range(len(anom_scores))))
        anoms_only = list(filter(lambda x:  anom_scores[x] == 1, range(len(anom_scores))))

        if len(anoms_only) < len(safe_only):
            safe_sample_indices = random.sample(safe_only, round(len(anoms_only)))
        else:
            safe_sample_indices = safe_only
        sample_preds = []
        sample_truths = []

        for i in range(0, len(safe_sample_indices)):
            sample_preds.append(anom_pred_scores[safe_sample_indices[i]])

        for i in range(0, len(anoms_only)):
            sample_preds.append(anom_pred_scores[anoms_only[i]])

        for i in range(0, len(safe_sample_indices)):
            sample_truths.append(anom_scores[safe_sample_indices[i]])

        for i in range(0, len(anoms_only)):
            sample_truths.append(anom_scores[anoms_only[i]])

        x_test = scaler.transform(x_test.reset_index(drop=True))
        sample_values = pd.DataFrame()
        count = 0
        for i in range(0, len(safe_sample_indices)):
            sample_values[count] = (x_test[safe_sample_indices[i]])
            count += 1

        sample_values = sample_values.transpose()
        sample_std = np.std(sample_values)

        count = 0
        full_values = pd.DataFrame()
        for i in range(0, len(safe_only)):
            full_values[count] = (x_test[safe_only[i]])
            count += 1

        full_values = full_values.transpose()
        full_std = np.std(full_values)

        full_anom_scores = anom_scores
        full_pred_scores = anom_pred_scores

        anom_scores = sample_truths
        anom_pred_scores = sample_preds

        # sample_loss = [0,0,0,0,0,0,0,0,0,0]
        # sample_times = [0,0,0,0,0,0,0,0,0,0]

        # test_loss= pd.DataFrame(test_loss).reset_index()
        # timepoints= pd.DataFrame(timepoints).reset_index()

        # for i in range(0, len(safe_sample_indices)):
        #     sample_loss = sample_loss.append(test_loss['index'][safe_sample_indices[i]])
        #     sample_times = sample_times.append(timepoints['%time'][safe_sample_indices[i]])

        # for i in range(0, len(safe_sample_indices)):
        #     sample_loss = sample_loss.append(test_loss['index'][anoms_only[i]])
        #     sample_times = sample_times.append(timepoints['%time'][anoms_only[i]])

        # print("Sample TN Size: %d "%len(sample_values))
        # print("Sample (Full) TP Size: %d "%len(anoms_only))

        # Flight performance metrics
        acc_score = accuracy_score(anom_scores, anom_pred_scores)
        print("Accuracy =  %0.4f" % accuracy_score(anom_scores, anom_pred_scores))

        prec_score = average_precision_score(anom_scores, anom_pred_scores)
        print("Precision = %0.4f" % average_precision_score(anom_scores, anom_pred_scores))

        auc = roc_auc_score(anom_scores, anom_pred_scores)
        print("AUC of ROC = %0.4f" % roc_auc_score(anom_scores, anom_pred_scores))

        recall = recall_score(anom_scores, anom_pred_scores)
        print("Recall = %0.4f" % recall)

        x_test = pd.DataFrame(x_test)
        # Plot losses over time with anomaly threshold line
        plt.scatter(x=x_test.index, y=test_loss, s=1, c=full_anom_scores)
        plt.plot(range(len(threshold)), threshold, c="red")
        # plt.axhline(threshold, color="red")
        plt.title(label=fid[0])
        plt.show()

        x_test = pd.DataFrame(x_test)
        # Plot losses over time with anomaly threshold line
        plt.scatter(x=x_test.index, y=test_loss, s=1, c=full_anom_scores)
        plt.axhline(y=static_threshold, xmin=0, xmax=510, color="red")
        plt.title(label=fid[0])
        plt.show()

        matrix = confusion_matrix(anom_scores, anom_pred_scores)

        print("Confusion Matrix:")
        print(matrix)
        print('\n')

        # Print boxplot for variable distributions

        # fv_list = list(np.mean(full_values))
        # sv_list = list(np.mean(sample_values))

        # bp_df = pd.DataFrame()
        # bp_df["Full Flight"] = fv_list
        # bp_df["Sample"] = sv_list

        # plt.boxplot(bp_df)
        # plt.show()

        if fid[1] == "engines":
            engine_acc_log.append(acc_score)
            engine_prec_log.append(prec_score)
            engine_auc_log.append(auc)
            engine_recall_log.append(recall)
            # engine_sample_dp.append(sample_values)
            # engine_full_dp.append(full_values)

    # Average performance metrics over all x_test flights
    print('\n\n')
    print("Engine Failures:")
    print("Average AUC = %0.3f (std. dev = %0.2f)" % (np.mean(engine_auc_log), np.std(engine_auc_log)))
    print("Average Accuracy = %0.3f (std. dev = %0.2f)" % (np.mean(engine_acc_log), np.std(engine_acc_log)))
    print("Average Precision = %0.3f (std. dev =  %0.2f)" % (np.mean(engine_prec_log), np.std(engine_prec_log)))
    print("Average Recall = %0.3f (std. dev = %0.2f)" % (np.mean(engine_recall_log), np.std(engine_recall_log)))

    list1 = [np.mean(engine_auc_log), np.std(engine_auc_log), np.mean(engine_acc_log), np.std(engine_acc_log),
             np.mean(engine_prec_log), np.std(engine_prec_log), np.mean(engine_recall_log), np.std(engine_recall_log)]
    excel_list = np.vstack((excel_list, list1))

data = pd.DataFrame(excel_list)
data.to_excel('model_comparison.xlsx', sheet_name='Dynamic Thresholding with DW', index=False)

