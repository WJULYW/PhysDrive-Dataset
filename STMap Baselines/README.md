You need to first preprocess the video following this [repo](https://github.com/WJULYW/HSRD/tree/main/STMap) and prepare your dataset. Then you can run these code by modifying the dataset address in these files:
1. cross_test.py: Evaluate deep-learning methods by scenarios.
2. intra_train.py: Train the supervised deep-learning methods.
Please note that, you need to modify the utils.py or MyLoss.py for adapting different models.
