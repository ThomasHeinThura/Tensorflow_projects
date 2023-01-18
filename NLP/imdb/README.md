These are models that train on the IMDB_review dataset.


|models                   |val_accuary    |val_loss       |time       |epochs         |
|-------------------------|---------------|---------------|-----------|---------------|
|imdb_hub_conv            | 87.22%        |0.3542         |2min54sec  |10             |
|imdb_hub_dense           | 87.02%        |0.3323         |2min32sec  |10             |
|imdb_USE                 | 85%           |0.3274         |3min38sec  |10             |
|imdb_hub_lstm            | 83%           |0.5017         |2min43sec  |10             |
|imdb_conv                | 86.11%        |0.3410         |1min40sec  |10             |
|Navie bayes              | 85.89%        |???            |???        |??             |

Short Note from testing and building model
1. The bilateral LSTM takes a lot of RAM(double the LSTM model) and time to train. So I skip that because of the low ram on my laptop.
2. The training dataset needs to extract for prediction and data visualization.
3. Due to a lack of knowledge and skill, I take a lot of time to figure out how to extract the dataset, how to calculate the accuracy, how to fit, how to tokenize and vectorize, and how to make embedding.
4. In this dataset, I try two transfer-learning embedding layers(hub layers and USE layers). Both give very good results.
5. The Naive-Bayes model is in imdb_1.ipynb 
6. I built a feature extraction Bert transformer model and it can only run on GPU and need fine-tuning.

Further study: Need to figure out why RNN models are likely to overfit in their own tokenize and embedded models.