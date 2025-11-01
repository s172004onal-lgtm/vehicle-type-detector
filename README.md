# vehicle-type-detector
Vehicle Type Detection

Problem Statement
With the rapid increase in the number of vehicles on roads, automated systems for traffic management, parking control, and surveillance have become essential. One of the key capabilities of such systems is the ability to detect and classify vehicle types such as cars, buses, trucks, and bikes from images or videos. Manual identification is time-consuming and prone to errors, making it unsuitable for large-scale or real-time use.

Solution
This project develops an AI-based Vehicle Type Detection System that automatically classifies vehicles into predefined categories using a deep learning model. The system uses image data to train a Convolutional 
Neural Network (CNN) capable of learning the distinguishing visual features of different vehicle types. Once trained, the model can predict the type of vehicle in unseen images accurately and efficiently.

Model
The model used in this project is ResNet-18, a deep convolutional neural network pre-trained on the ImageNet dataset. The final fully connected layer was modified to match the number of vehicle classes in the dataset. Transfer learning was applied to fine-tune the network for better accuracy and faster convergence. The model was implemented using PyTorch, trained with the CrossEntropyLoss function, and optimized using the Adam optimizer.

Approach
Dataset Loading – The "aryadytm/vehicle-classification" dataset from Hugging Face was used.
Data Preprocessing – All images were resized to 224x224 pixels and normalized using ImageNet mean and standard deviation values.
Model Setup – A pre-trained ResNet-18 model was loaded and modified for multi-class classification.
Training – The model was trained for three epochs using an 80–20 train-validation split and a batch size of 16.
Evaluation – Training performance was monitored using loss values, and the trained model weights were saved as "vehicle_model.pth".
Deployment – The trained model can be used for predicting vehicle types from new images or extended for real-time detection using OpenCV.
