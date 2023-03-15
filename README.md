# Bird Species Classification using ResNet18 and Fine-tuning with Interpretability using LIME
This project is focused on training a deep learning model to classify 500 different species of birds. The model used in this project is ResNet18, which has been fine-tuned using transfer learning to improve its accuracy. Additionally, interpretability has been incorporated into the model using the LIME library.

## Dataset
The dataset used in this project is the CUB-200-2011 dataset, which contains 11,788 images of 200 bird species. Each species has around 30 images, making it a highly challenging dataset. The dataset has been split into 70% training, 15% validation, and 15% testing sets.

## Model Architecture
The ResNet18 model was used as the base model, which was pre-trained on the ImageNet dataset. The final layer of the model was replaced with a fully connected layer with 500 output units, representing the 500 bird species. The model was trained using the fine-tuning approach, where only the final layer's weights were updated during training. The model was trained for 10 epochs with a batch size of 128 and an initial learning rate of 0.001.

## Interpretability
Interpretability was incorporated into the model using the LIME (Local Interpretable Model-Agnostic Explanations) library. LIME was used to generate explanations for the model's predictions, allowing us to understand the model's decision-making process better. LIME generates a heatmap that highlights the most important parts of an image that the model used to make its prediction.

## Results
The trained model achieved a test loss of 0.198 and a test accuracy of 95.04%. The interpretability features of the model showed that the model was making its predictions based on specific features of the birds, such as the beak's shape, the color of the feathers, and the size of the bird.

### Classification Report
```
                 precision    recall  f1-score   support
       accuracy                           0.95      2500
      macro avg       0.96      0.95      0.95      2500
   weighted avg       0.96      0.95      0.95      2500
```

## Files
- `notebooks/`: The folder containing Jupyter notebooks used in the project.
- `models/`: The folder containing the trained model weights and architecture. The best model is saved here.
- `README.md`: The readme file for the project.

## Conclusion
This project showcases the effectiveness of fine-tuning pre-trained models for challenging classification tasks. Additionally, incorporating interpretability into the model helps to understand the decision-making process better, which can be useful for debugging and improving the model.
