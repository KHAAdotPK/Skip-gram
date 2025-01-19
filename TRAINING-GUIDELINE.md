# Skip-gram Model Training Guidelines

## 1. Learning Rate Configuration

* **Initial Learning Rate**: 0.025
* Higher initial rate compensates for absence of negative sampling

* **Learning Rate Decay**:
* Multiply learning rate by 0.99 after each epoch

* **Justification for Higher Learning Rate**:
* Small vocabulary size
* Simple and distinct phrases
* No regularization or negative sampling implementation

## 2. Epoch Settings

* **Recommended Range**: 50-75 epochs

* **Reasoning**:
* Suitable for small dataset size
* Simple word relationships in data
* Faster convergence due to absence of negative sampling
* Monitor loss function for early convergence possibility

## 3. Additional Training Tips

* **Window Size**:
* Use 2-3 words window size
* Optimal for given phrase structure

* **Weight Initialization**:
* Initialize between -0.5/dim and 0.5/dim
* dim = embedding dimension

* **Early Stopping**:
* Implement if loss doesn't improve for 5-10 consecutive epochs

## Note
These parameters are optimized for a medical symptoms dataset with approximately 50 phrases and simple word combinations. The higher learning rate with decay helps compensate for the lack of negative sampling, while the moderate epoch count prevents overfitting on the small dataset.

