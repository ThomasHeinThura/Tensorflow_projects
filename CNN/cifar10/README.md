These are models that train on the cifar10 dataset.


|models                  |val_accuracy    |val_loss       |time       |epochs         |
|------------------------|----------------|---------------|-----------|---------------|
|cifar10_gpu             | 89%            |0.4762         |36min14sec |50(41-early)   |
|cifar10_conv_best       | 86%            |0.4491         |53min      |25             |
|cifar10_another_best    | 78%            |0.6680         |11min10sec |50(37-early)   |
|cifar10_dense           | 56%            |1.515          |17min27sec |25             |
|cifar10_conv            | 63% (f1:10%)   |1.1963         |6min52sec  |25             |

Short Note from testing and building model
1. val_accuracy is just a number that looks great to confirm these numbers are accurate then first look at val_loss values if val_loss is over 1 or 0.8 that means the Model is likely to overfit. your prediction values are likely to get lower values. 
2. always calculates f1 and accuracy scores to confirm your models are not overfitting. Try not to satisfy with just the val_accuracy score being high.
3. dropout and regulation of the model are as important as preparing the data. 

Further study: Try to test the dataset with transfer learning. I don't get any luck with the Features Extraction EfficientnetB0 model. I found out some Renet50v2 models in Kaggle get 90+% val_accuracy.
