You need to first preprocess the raw mmWave data following this [repo](https://github.com/WJULYW/PhysDrive-Dataset/tree/main/mmDG-prepocess) and prepare your dataset. Then you can run these code by modifying the dataset address in these files:

cross_test.py: Evaluate deep-learning methods by scenarios.
intra_train.py or intra_train_IQMVED.py: Train the supervised deep-learning methods. Please note that, you need to modify the hyper-parameter -m for adapting different models.