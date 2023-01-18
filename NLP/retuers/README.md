|models                   |val_accuary    |val_loss       |time       |epochs         |
|-------------------------|---------------|---------------|-----------|---------------|
|reuters_dense            | 81.45%        |0.8064         |30sec      |5              |
|reuters_conv1D           | 81.02%        |0.9033         |3min20sec  |5              |
|reuters_USE              | 80.45%        |0.8013         |1min45sec  |15             |
|reuters_hub              | 79.79%        |0.9912         |5min50sec  |10             |
|Navie bayes              | 59.97%        |???            |???        |??             |


* This dataset is a bit tricky to do because input data is tokenized and the training of the dataset is likely to overfit easily.
* In this dataset, only conv1D and dense models are easy to train for 3-10min.
* The others RNN models that a lot of time 5min for one epoch and take a lot of memory. I think it is because when the dataset input is tokenized and the training starts with a large tensor array and takes heavy input to RNN layers.
* From my point of view, the main reason the model is easy to overfit and val_loss is hard to come down to 0.8 mark is because of the dataset.
* Trying to import from the text dataset or reverse the tokenized data to the text data and embedded again may fix the problem. 
* I try reversing the tokenized data to the text data and the RNN model can be trained easily.
* Although the RNN model can be trained easily, val_accuracy is not increased much and is likely to overfit.
