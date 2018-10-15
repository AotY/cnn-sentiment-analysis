## CNN For Sentiment Classification

### Model 
Model overview:

    ```
    CSC(
      (embedding): Embedding(5481, 300, padding_idx=0)
      (dropout): Dropout(p=0.8)
      (conv1): Conv2d(1, 256, kernel_size=(3, 300), stride=(1, 1))
      (conv2): Conv2d(1, 128, kernel_size=(3, 256), stride=(1, 1))
      (fc1): Linear(in_features=128, out_features=32, bias=True)
      (fc2): Linear(in_features=32, out_features=2, bias=True)
      (softmax): LogSoftmax()
    )
    ```

### data
- Movie Review Data: http://www.cs.cornell.edu/people/pabo/movie-review-data/

### requirement
- Pytorch (0.4 +)

