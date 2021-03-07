# FaceMaskDetection
In an attempt to bring more transparency in artificial intelligence in a high stakes situation such as the Coronavirus pandemic, our aim was to create a model that would be able to determine if an individual was wearing a mask correctly, incorrectly, or not at all. Utilizing a subsection of the dataset MaskedFace-Net, we were able to train a model with the Inception Resnet V1 model. Moreover, as this dataset further breaks down incorrect mask usage into why, such as uncovered chin, mouth, or nose area, we aimed to apply GradCAM in order to build transparency and trust, and ultimately ensure that our model was coming to the conclusion for the right reasons.

In order to run out code on a given input path, type the command ```python run.py test``` from the maain directory. 
This calls the 

The presentations section includes our model metrics for the neural network created for our analysis. This goes on to look at our accuracy metrics and see how the model performed. Our notebook GradCam EDA looks at implementing an algorithm that can identify what our neural network would look at to identify whether a mask would be work correctly. An example of that is the output of our run.py file which gives us information on how these images are segregated. The src folder contains information on the functions used to train the model and our config folder contains parameter information that simplifies the working of run.py. 

In order to run our given code structure, the command python run test needs to be applied. What this does is load data using the data_params mentioned, and identify certain key aspects needed for our model's analysis. 
