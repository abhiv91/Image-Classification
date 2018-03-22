# Comparing the performance of CNNs with traditional ML algorithms for image recognition given small sample sizes

This is the repo for my experiment where I measured the performance of the hot new image classifying model developed by Yann LeCun at Facebook, Convolutional Neural Networks, with a Support Vector Classifier (SVC). Read the paper to learn more about transfer learning and leveraging Google's Inception CNN. 

The main script used in this project is DataCompetition.py and the other scripts are helper functions for the SVC or for data augmentation. I followed the Tensorflow Inception repo found here for much of the training of the model, only modifying the final layer with my training set:  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py 
