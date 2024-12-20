# 4AL3-final-project
Final Project for SFWRENG 4AL3

This project trains a ML model to identify noise artifacts on fundus images.

The code is in this repo. 
The Training.py file loads the dataset, preprocesses it, and trains the model.
The Test.py file loads a test dataset, loads the model file, and uses it on test dataset to generate predictions and display them. 

The full training and validation dataset is in Fundus_FullDataSet.
A smaller dataset is in Fundus_PartialDataSet.

The test images are in Fundus_TestDataSet.

The datasets were generated using eyedata/eyedata.py on images from the Diabetic Retinopathy Detection dataset on Kaggle: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data. The specific data used from this dataset is in Fundus_SourceDataSet.

To set up the enviornment, use Python 3.11, and use the requirements.txt.

To run Training.py (using the partial dataset):

   cd eyedata
   git clone Fundus_PartialDataSet
   cd ..
   python Training.py eyedata/Fundus_PartialDataSet

To run Test.py (using the submitted model file):

   cd eyedata
   git clone Fundus_TestDataSet
   cd ..
   python Test.py model_file eyedata/Fundus_TestDataSet

To use a different model file, specify, eg: 
   python Test.py models/model_file_20241216-231644_final_torch eyedata/Fundus_TestDataSet
