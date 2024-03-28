# INLP-Assignment-3
`
Sanika Damle
2021115005
`

## SVD
For generating the embeddings using svd run the following command:
```
python3 svd.py
```
Enter window size (best performing window size is 2)
The embeddings will be saved as 
`f'./models/svd-vocab_{str(window_size)}.pt'`

For running the LSTM (news classification)
run 
```
python3 svd-classification.py
```
There is an option to load a pretrained model or train it from scratch. The context size is also taken as input and metrics on the test set are displayed. The confuson matrix is stored as 
`f'./models/svd-classification-model_{str(window_size)}.pt'`


## Skip Gram
For generating the embeddings using skip gram run the following command:
```
python3 skip-gram.py
```
Enter window size (best performing window size is 2)
The embeddings will be saved as 
`f'./models/skip-gram-vocab_{str(window_size)}.pt'`

For running the LSTM (news classification)
run 
```
python3 skip-gram-classification.py
```
There is an option to load a pretrained model or train it from scratch. The context size is also taken as input and metrics on the test set are displayed. The confuson matrix is stored as 
`f'./models/skip-gram-classification-model_{str(window_size)}.pt'`