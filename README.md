# Image-analysis-neural-network
## Project Introduction
This project utilizes the dataset from the expired Kaggle competition [Dogs vs. Cats Redux: Kernels Edition](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/overview) for deep learning exploration.

## Methodology
The approach involves experimenting with both a conventional Convolutional Neural Network model and a pre-trained ResNet101 model from Keras. The goal is to assess performance differences and refine the process of tuning deep neural networks.

## Traditional Convolutional Neural Network
I used a traditional Convolutional Neural Network model for the prediction. I utilized all images from the training dataset to train the model. The images were resized into 256 rows, 256 columns, and 3 channels arrays. Since the proportion of dogs and cats images in the training dataset was well balanced, there was no imbalance issue in this case.

The model comprised two main components: convolution layers and dense layers. In the initial phase, eight convolution layers were employed to capture essential patterns from the input images. MaxPooling layers were incorporated to effectively reduce the number of parameters. To address the issue of vanishing or exploding gradients, batch normalization was applied to every second convolution layer. The activation function used was ReLU, chosen to counteract the problem of gradient vanishing and promote optimal feature learning within the convolutional layers.

Moving on to the second phase, a single dense layer with 512 neurons was introduced, accompanied by batch normalization to enhance training stability. Observing signs of overfitting in the validation and training loss line chart, an early stop mechanism was implemented during the training process. Additionally, to mitigate overfitting, the number of training epochs was reduced to 20, and the batch size was 50. 

The final score is 0.46018. 

## Keras Pretrained Model ResNet101
I utilized the pre-trained ResNet101 model from Keras for prediction. The training process involved using 80% of the data from the training dataset for training the model, while the remaining 20% served as the validation dataset. To accommodate the model's limitations, I assumed uniform dimensions for each image, resizing them all to a 160x160x3 array.

The model was structured with three main components: the ResNet, a pooling layer, and an output layer. The ResNet101 layer was initialized with hyperparameters 'weights=imagenet,' indicating the use of pre-trained weights from the ImageNet dataset. Additionally, the hyperparameter 'include_top' was set to False, allowing for the addition of a custom output layer later on. To optimize computational efficiency, I did not add any larger Dense layer. I opted for GlobalAveragePooling2D() instead of Flatten() after the ResNet layer since Global pooling consolidates all feature maps into a single map, simplifying the information for a subsequent dense classification layer. Finally, the output layer was configured as a Dense layer with a single neuron, employing the 'sigmoid' activation function.

The training configuration included setting the number of epochs to 10 and the batch size to 128 to prevent overfitting during the training process.

The final score is 0.11526.
