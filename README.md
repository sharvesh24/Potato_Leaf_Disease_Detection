# Potato Leaf Disease Detection


Plant diseases pose a significant challenge to the agricultural industry, with early identification playing a crucial role in managing infections and improving crop yield. This study focuses on developing a deep learning-based model for detecting potato leaf diseases. The model classifies potato leaves into three categories: healthy, early blight, and late blight. To enhance training effectiveness, the dataset is split into an 80/20 ratio, ensuring a balanced learning process. Various data augmentation techniques, such as flips and rotations, are applied to prevent overfitting and improve model generalization.


For classification, a convolutional neural network (CNN) is implemented with the Adam optimizer, known for its adaptive learning rate and efficient convergence. The model is trained on a dataset of labeled potato leaf images, allowing it to extract key features and make accurate predictions. To further enhance performance, hyperparameter tuning and regularization techniques are utilized, ensuring optimal accuracy while reducing computational overhead.


The model achieves a high accuracy of 97%, demonstrating its effectiveness in identifying potato leaf diseases. To validate the performance, visualization techniques such as confusion matrices and accuracy/loss graphs are generated. These visualizations provide insights into the model’s classification capabilities, highlighting precision, recall, and overall effectiveness. Comparative analysis with other machine learning techniques, such as support vector machines (SVM) and random forests, further establishes the CNN’s superiority in disease detection.


This study presents a robust and efficient deep learning approach for automated plant disease detection, which can assist farmers in early disease identification and timely intervention. The integration of deep learning and image processing in agriculture opens new possibilities for scalable and accurate plant disease monitoring systems, ultimately contributing to improved agricultural productivity and sustainability.
