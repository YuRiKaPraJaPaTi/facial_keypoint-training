
## Model Modifications

The facial keypoint detection model is based on the VGG16 architecture, with the following modifications:

1. **Frozen Convolutional Layers**: All parameters in the convolutional layers are set to not require gradients. This means the pre-trained weights from ImageNet will remain unchanged during training, allowing the model to leverage learned features effectively.

2. **Custom Average Pooling Layer**: The original average pooling layer has been replaced with a new sequence that includes:
   - A convolutional layer (`Conv2d`) to further refine feature extraction.
   - A max pooling layer (`MaxPool2d`) to reduce the spatial dimensions.
   - A flattening layer to convert the 3D output into a 1D tensor.

3. **Modified Classifier**: The classifier is restructured to:
   - Include a fully connected layer (`Linear`) that outputs to 512 neurons, followed by a ReLU activation.
   - Introduce a dropout layer to minimize overfitting.
   - End with a final linear layer that predicts 136 outputs (representing the coordinates of 68 facial landmarks) using a Sigmoid activation function, ensuring output values range from 0 to 1.

These modifications tailor the VGG16 architecture specifically for the task of detecting facial keypoints.

Below is the code used to implement these modifications:


```python
def get_model(device): 
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for layers in model.parameters():
        layers.requires_grad = False  # No gradient descent work, layers do not optimize

    model.avgpool = nn.Sequential(
        nn.Conv2d(512, 512, 3),
        nn.MaxPool2d(2),
        nn.Flatten()
    )
    model.classifier = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, 136),
        nn.Sigmoid()  # Sigmoid for point probabilities (0 to 1)
    )
    return model.to(device=device)
```




