
These are models that train on the flower dataset.


|models                  |val_accuary    |val_loss       |time       |epochs         |
|------------------------|---------------|---------------|-----------|---------------|
|flower_best             | 81%           |0.6898         |16min46sec |25             |
|flower_conv             | 78%           |0.6780         |14min19sec |25(23-early)   |
|flower_vgg19            | 76.01%        |0.6437         |1hr9min4sec|50(26-early)   |
|flower_dense            | 42%           |1.3371         |3min1sec   |10             |
|flower_gpu              | ~75%          |~0.77          |~25min     |50(30-early)   |

Short Note from testing and building model
* The training on GPU is challenging because the graphic RAM is easily full. So,
    1. Need to reduce batch size and 
    2. Need to reduce the usage of complex network
* If you load data from the TensorFlow dataset with batch_size = -1 and do data augmentation, it is likely to fill your RAM and crush the system.
* The dataset is need to be fixed. There are a lot of photos which is not true to their labels. Example. in the rose dataset, there are human photos that are no roses.
* It is hard to get over 85% accuracy with a base Conv2D model. Need to try fine-tuning models.  
