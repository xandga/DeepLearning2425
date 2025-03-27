# DeepLearning2425

## Important Notes:

**Option 1: Predict Phylum First, Then Family**

* * We can first predict the phylum and then predict the family based on the predicted phylum.
  * This approach may require different preprocessing steps for each phylum, depending on the species within them.
  * The images will need to be divided and processed based on their phylum and family before training.
* **Option 2: General Preprocessing for All Classes**
  * We can treat the task as a classification problem for both phylum and family directly without separating them.
  * This method will involve general preprocessing like resizing and cropping the images to a standard size before training.
  * The preprocessing will be uniform for all images, regardless of their phylum or family.

This decision will affect how we approach the preprocessing and model structure for our task.

## **19/03 Class**

### Project Tips

- some images are not in RBG, so we need to transform it - [Image data loading](https://keras.io/api/data_loading/image/)
- using image_dataset_from_directory we can select the image_size (cuz there diff sizes) or explore first the sizes of teh images and then resize - [Image data loading](https://keras.io/api/data_loading/image/)
- look also to this [Image data loading](https://keras.io/api/data_loading/image/) for preprocessing

**using the resizing that keras provided or with the image data loading**

* other problem that we will have *overfitting*: we can try data augmentation [Data augmentation  |  TensorFlow Core](https://www.tensorflow.org/tutorials/images/data_augmentation) or [Image augmentation layers](https://keras.io/api/layers/preprocessing_layers/image_augmentation/)
* we can use random zoom for data augmentation
* change the darkness/brigthness

⚠️exagerating the data augmentation can destroy the patterns that we need for the model to learn

* try to use some [Keras Applications](https://keras.io/api/applications/)  - which provide pre-trained models that can be fine-tuned for your task. Since these models are already trained, update the weights **gradually** to preserve learned features. Additionally, **reduce the learning rate** to avoid overwriting existing knowledge while allowing the model to adapt to your dataset.

### Extra Notes (not directly related with the Project)

* Training a model from scratch can lead to high variability, making it harder to converge effectively.
* Adding more layers can worsen the issue, as **nonlinear transformations** can introduce instability, especially when applied to **initially linear inputs** resulting in a highly nonlinear output.
* One way to address this challenge is by using a  **Residual Neural Network (ResNet)** , which helps maintain stable learning by allowing the model to learn residual mappings rather than full transformations.

## **26/03 Class**

* leaky relu vs relu
* Gated recurrent unit
