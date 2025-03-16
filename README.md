# DeepLearning2425

Important Notes:


#IMPORTANT NOTES

* **Option 1: Predict Phylum First, Then Family**
  * We can first predict the phylum and then predict the family based on the predicted phylum.
  * This approach may require different preprocessing steps for each phylum, depending on the species within them.
  * The images will need to be divided and processed based on their phylum and family before training.
* **Option 2: General Preprocessing for All Classes**
  * We can treat the task as a classification problem for both phylum and family directly without separating them.
  * This method will involve general preprocessing like resizing and cropping the images to a standard size before training.
  * The preprocessing will be uniform for all images, regardless of their phylum or family.

This decision will affect how we approach the preprocessing and model structure for our task.
