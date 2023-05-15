# Crowd-Counting-with-Self-training
This is the codes for my Undergraduate Graduation Thesis named "Semi-supervised Crowd Counting Based on Consistency Regularization and Iterative Retraining".

This work is based on St++, a method for semi-supervised semantic segmentation.

> Yang L, Zhuo W, Qi L, et al. St++: Make self-training work better for semi-supervised semantic segmentation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 4268-4277.

## Abstract
Crowd counting aims to calculate the number of people in a given image or a video with computer algorithms, being widely used in public safety management, intelligent monitoring, and other fields. Based on the data-intensive deep learning, it is extremely tough to label images in crowd counting (labels are density maps, and all heads need to be manually labeled), leading to limited labeled data. Therefore, this paper considers using a semi-supervised setting to enhance model performance with a large amount of unlabeled data.

Semi-supervised learning based on self-training can use unlabeled data effectively, but the simple self-training framework can not be applied to crowd counting well. Without preprocessing pseudo-labels using methods like one-hot encoding before retraining the crowd-counting model using unlabeled data, no new semantic information will be introduced to the retraining stage.

To this end, this paper introduces strong-weak consistency regularization, which provides new semantic information and alleviates the over-fitting problem of the model, significantly improving the generalization ability of the model.

Furthermore, this paper also adopts iterative retraining. Different from the simple self-training framework that uses all (including unreliable) unlabeled data at the same time, iterative retraining iteratively adds the most reliable unlabeled data to the training set. The experiments show that iterative retraining can improve the prediction ability of the model while combining it with consistency regularization.
