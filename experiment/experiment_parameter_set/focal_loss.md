```
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      3479
           1       0.47      0.43      0.45        67

    accuracy                           0.98      3546
   macro avg       0.73      0.71      0.72      3546
weighted avg       0.98      0.98      0.98      3546
```

$$\;with\;\alpha=0.5,\;\gamma=2.0$$
But we can notice that the 1 sample is 100 times smaller thant 0 sample

So we can try with $$alpha=0.5\;\gamma=3.0$$

or

$$alpha=0.75\;\gamma=2.0$$

which means the weight of 1's sample has 0.75 and the weight of 0's sample is 1-0.75

$$\gamma=3.0$$

```
Best Threshold: 0.120, G-Mean: 0.773
Shape of flattened labels: (4096,)
Shape of flattened predictions: (4096,)
              precision    recall  f1-score   support

           0       0.98      0.85      0.91      3439
           1       0.21      0.70      0.32       192

    accuracy                           0.84      3631
   macro avg       0.59      0.77      0.61      3631
weighted avg       0.94      0.84      0.88      3631
```

$$ try \;\alpha=0.97, \gamma=3.0$$

```
Best Threshold: 0.118, G-Mean: 0.726
Shape of flattened labels: (4096,)
Shape of flattened predictions: (4096,)
              precision    recall  f1-score   support

           0       0.99      0.71      0.83      3571
           1       0.05      0.73      0.09        71

    accuracy                           0.71      3642
   macro avg       0.52      0.72      0.46      3642
weighted avg       0.97      0.71      0.81      3642
```

$$\alpha=0.97, \gamma=2.0$$
```
Best Threshold: 0.114, G-Mean: 0.817
Shape of flattened labels: (4096,)
Shape of flattened predictions: (4096,)
              precision    recall  f1-score   support

           0       0.99      0.89      0.94      3257
           1       0.22      0.74      0.33       131

    accuracy                           0.89      3388
   macro avg       0.60      0.82      0.64      3388
weighted avg       0.96      0.89      0.91      3388
```

