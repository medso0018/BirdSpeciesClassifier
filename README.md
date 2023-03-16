# Bird Species Classification using ResNet18 and Fine-tuning with Interpretability using LIME
This project is about training a deep learning model to classify 500 different species of birds using ResNet18, which has been fine-tuned using transfer learning to improve its accuracy. The project also incorporates interpretability into the model using the LIME library. The dataset used for training, testing, and validation consists of 80,085 training images, 2,500 test images, and 2,500 validation images (5 images per species) of high-quality original images. This project was completed as part of the [Kaggle challenge on bird species classification](https://www.kaggle.com/datasets/gpiosenka/100-bird-species).

## Dataset
The dataset consists of 500 bird species with 80,085 training images, 2,500 test images, and 2,500 validation images (5 images per species). Each image only contains one bird and typically takes up at least 50% of the pixels in the image, resulting in high-quality images. All images are in JPEG format and have a resolution of 224 x 224 x 3. The dataset includes a train set, test set, and validation set, each containing 475 subdirectories (one for each bird species). The dataset also includes a CSV file (birds.csv) with information on the file paths, labels, scientific labels, data set designation, and class index values for each image. The data structure is convenient for use with the Torchvision ImageFolder function to create data generators.

## Model Architecture
To optimize the training process quickly, the model used a pre-trained ResNet18 architecture as the base model, which had already learned features from a large dataset, the ImageNet. A fully connected layer with 500 output units was added to the pre-trained model to predict 500 bird species. The entire model was trained using the backpropagation algorithm with the categorical cross-entropy loss function and optimized using the Adam optimizer. During the training process, all the layers and parameters in the model, including the weights in the added fully connected layer and the pre-trained ResNet18 model, were updated to minimize the loss function. The Adam optimizer was used to adjust the learning rate during training for efficient convergence. The model was trained for 10 epochs with a batch size of 128 and an initial learning rate of 0.001.

## Interpretability
Interpretability was incorporated into the model using the LIME (Local Interpretable Model-Agnostic Explanations) library. LIME was used to generate explanations for the model's predictions, allowing us to understand the model's decision-making process better. LIME generates a activation map (Positive/Negative) that highlights the most important parts of an image that the model used to make its prediction.

## Results
The trained model achieved a test loss of 0.198 and a test accuracy of 95.04%. The interpretability features of the model showed that the model was making its predictions based on specific features of the birds, such as the beak's shape, the color of the feathers, and the size of the bird.

### Classification Report
```
                 precision    recall  f1-score   support
       accuracy                           0.95      2500
      macro avg       0.96      0.95      0.95      2500
   weighted avg       0.96      0.95      0.95      2500
```

## Files & Directories
- `notebooks/`: The folder containing Jupyter notebooks used in the project.
- `models/`: The folder containing the trained model weights and architecture. The best model is saved here.
- `README.md`: The readme file for the project.

## Conclusion
This project showcases the effectiveness of fine-tuning pre-trained models for challenging classification tasks. Additionally, incorporating interpretability into the model helps to understand the decision-making process better, which can be useful for debugging and improving the model.
