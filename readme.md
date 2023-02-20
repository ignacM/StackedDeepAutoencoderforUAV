### Steps:
1. Download "processed.zip" data from https://kilthub.cmu.edu/articles/dataset/ALFA_A_Dataset_for_UAV_Fault_and_Anomaly_Detection/12707963 and unzip in working directory.
2. Install The Necessary Libraries: pip install -r requirements.txt<br> 
3. Use [main.py](main.py) to run the StackedAutoencoder with dynamic thresholding and weight loss function. 
4. Open [Dyanmic_Thresholding.xlsx](Dyanmic_Thresholding.xlsx) to see results


An autoencoder reconstruction without weight loss function can be used by replacing [train_SAE.py](train_SAE.py) with
[train_SAE_without_DWL.py](bin%2Ftrain_SAE_without_DWL.py).


### Please check https://doi.org/10.4271/01-15-02-0017