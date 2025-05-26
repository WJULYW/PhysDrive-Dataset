After preparing the dataset in as same way as [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox), you can run this code by modifying the dataset address in these files:
1. cross_traditional_test.py: Evaluate traditional methods.
2. cross_test.py: Evaluate deep-learning methods by scenarios.
3. sup_video_intra_train_baseline.py: Train the supervised deep-learning methods.
4. unsup_video_intra_train_baseline.py: Train the unsupervised deep-learning methods.

Please note that, since different methods have varied input sizes, you need to refer to their source papers and change the MyDataset.py or MyLoss.py.

Or, you can use this dataset by [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox).
