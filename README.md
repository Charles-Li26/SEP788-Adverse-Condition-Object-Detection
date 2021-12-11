# SEP788-Adverse-Condition-Object-Detection
1.	Problem Statement 
This project is aims to detect object and classified in adverse environment and variety of view. Through YOLO model to train the images during the adverse environment to realize real-time image testing and live video testing. 
The adverse environment should include rainy days, snowy days, dark nights and heavy sunshine. Which the camera will be covered with water and snowflake, or sight blocked cause by the darkness and strong sunlight. 
The critical problem for this project is to complete the classification and training the data set. This project will use Neural network to completer image pattern recognition, extract the image features and select the most important and critical features of the vehicle during the adverse environment into the neural network for training and recognition. 
Neural network can process and categorize images quickly and solve the feature binding problems by add more layers. It is convenient to operate and run, which is very suitable for this project.
2.	Understanding
As live detection is performed in adverse conditions, the training process would be slightly different from training the clear images on the roads. To enable fast and accurate real-time detection, YOLO with the effective detecting algorithm comes to a good option to perform the task. Although YOLO has remarkable performance in detecting objects in clear weather, the performance in adverse condition of the model is still doubtful. To address this problem, some image recovery tools might be needed to transfer the images in severe conditions to the normal conditions (e.g., recovering the pixels under snow and the pixels in the dark). By adding the model, the YOLO should be working fine with the images in clear conditions.  
The Canadian Adverse Driving Conditions dataset is selected to train for our object detection model as the datasets includes more than 70000 images (1280 x 1024) to train the model in the snow, rain and dark conditions. Also, the annotation file labeling the 3D bounding scan by lidar is attached as json file. The annotation file has labelled out class “truck”, “car”, “traffic guidance objects”, “Pedestrians” and the other key objects. Therefore, the annotation file is useful to train the model combining with the image dataset. 
While training the neural network, SoftMax is the most accurate activation function as it is useful to output layers while making multi-class predictions. The other key hyper parameters including dropout rate, learning rate, epochs, batch size also need to be considered to avoid over-fitting or underfitting the model. In this case, Tensor Board is going to be used to optimize those hyper parameters by visualizing the accuracy and loss at with different combinations of those hyper parameters.
3.	Data Pre-processing
Prior to the model training process, several data pre-processing steps are going to performance to ensure the trained model is not over-fitting or under fitting.  Those steps include importing the necessary library, data normalization, recovering the image as much as possible and splitting the dataset. As the dataset is constructed with various image files. The image pixels normalization is needed to divide each pixel on the image by 255 to minimize the pixel value between 0 to 1. In this case, the model is going to deal with those small pixels value effectively.  Image Inpainting is going to be performed ahead of training the model to reconstruct the missing regions in an image. The Inpainting may benefit a better training accuracy for the YOLO model. Finally, the dataset is going to be splitting into the training dataset, test, dataset, and validation dataset. The training data is going to be set as 75% of the total images data while the other 25% is going to be the test dataset. 80% of those 75% of the total images is set to be the training data and the other 20% is going to be the validation data. While the test data is going to test the model accuracy in predicting the dataset that the model has never seen (avoid underfitting), the validation data is going to test the model performance in predicting the familiar dataset (avoid overfitting).
4.	Approach to Implement
There are six procedures: 
Data Pre-processing: In the pre-processing step, we need to do data normalization task to reduce the pixel range from (0, 255) to (0, 1). In this way, we can eliminate the redundancy and inconsistent dependency. Then due to some images are covered by rain or snow and it will cause the NN model hard to identify the objects on that image. Therefore, “inpainting” as photo recovering method could be used in this project as image pre-processing.  
NN model Selection: As the dataset occupies 90GB storage so it is a big dataset. Therefore, it is hard to test various models’ accuracy and speed as it will take a long training time. Therefore, the team will do online research to discover the best model suitable for this project. In addition, the team will analyze if we need to develop a model by ourselves.  
Training: Due to the dataset has already labeled in 3D boundary box, so the team will analyze the needs to label in 2D box to speed up the inferring time. Then, there are some hyper parameters that need to define, such as epoch number, activation method, train and test split percentage, validation percentage. 
Testing: There are two testing method, one is image prediction to import a image to make the model to predict the object class. Another is live video prediction which is the hardest part, it will test the model classification speed and accuracy.  
Improvement: The final step is to find out if there are any wrong operation or miss any step to cause the accuracy and speed is not optimal. Then, the team needs to solve the problem to optimize the model.
5. Explanation of the Methods: 
5.1. Preprocessing
5.1.1. Attentive Generative Adversarial Network
 
The goal is to remove raindrops, thereby converting the raindrop image into a clean image. However, the problem is that part of the image is blocked by raindrops, and most of the background information in the blocked area is completely lost.
This network contains two networks, the first important section is the generative network, it will generate an attention map which will only focus on the raindrop area, it is produced by a set of ResNet combine with convolutional LSTM and some general convolutional layers. The second part is the autoencoder, it can take the input raindrop image and the attention map. Each convolutional layer can output a loss and it will compare with the ground truth, In the final output, it will apply the perceptual loss to obtain a more global similarity to the ground truth. 
Finally, in the discriminative network section, it can validate if the image is real or fake. However, as the background information lost as it is in the raindrop region, thus, it is hard for the discriminator to validate. So, it requires the attention map to guide the discriminator to focus on the miss region.
I = (1 − M) ⊙ B + R--------------------------------------------------------------------------------------------------------------(1)
 -----------------------------------------------------------------------------(2)
There is an equation that can represent the generative network. As you can see in equation (1), the I is the input image, M is the binary mask, B is the background image, R is the effect caused by the raindrops and reflection. As the raindrop is transparent and reflect some background information. The goal is to obtain the background image B from I by creatin an attention map. Then, in the discriminator formula as shown in equation (2), D is the discriminative network, G represents generative network, the discriminative network will validate the image. 
5.1.2. ResNet
The another Neural network to remove raindrop is to build a basic residual network with five ResBlock. The neural network model is listed below:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )
    def Resblock(self, input):
        
        z = input
        for i in range(6):
            
            z = torch.cat((input, z), 1)
            z = self.conv0(z)
            z = F.relu(self.res_conv1(z) + z)
            z = F.relu(self.res_conv2(z) + z)
            z = F.relu(self.res_conv3(z) + z)
            z = F.relu(self.res_conv4(z) + z)
            z = F.relu(self.res_conv5(z) + z)
            z = self.conv(z)
            z = z + input
            
        return z
In the first layer, it begins with a Conv+ReLU algorithm, then it follows with five ResBlocks and each block has Conv+ReLU algorithm and repeat two times. Finally, Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty. torch.cat() can be seen as an inverse operation for torch.split() and torch.chunk(). torch.cat() can be best understood via examples.
5.1.3. Image Lightness
Image enhancement is to making images more useful. In this profess the image will highlight the interesting detail and removed noise from the images which will making images more visually appealing. Generally, there are two categories of image enhancement, one is Spatial domain techniques, this technique could manipulate image pixels directly. Another one is frequency domain techniques; it manipulates Fourier transform or wavelet transform of an image. In this project we concentrate on the first technique.
In this project we use histogram equalization to process the image. The histogram of an image shows the distribution of grey levels in the image, and with high contrast image has the most evenly spaced histogram. The histogram equalization algorithm enhances the contrast of images by transforming the Values in an intensity image so that the histogram of the output image is approximately flat. Shows in the figure below. [1]
  
Figure 1 Histogram equalization
 
Figure 2 Histogram equalization sample code
5.1.4. Noise Removal
Erosion: all the pixels near boundary will be discarded depending upon the size of kernel
Dilation: Because erosion eliminates noise, but it also shrinks our objects. So we expand it. And dilation can be used to increase the object
MorphologyEX: has opening and closing function same as erosion and dilation.
The dataset is in adverse conditions; thus, it will create a lot of noises on the image. Therefore, morphological image processing is applied to remove the noises in the rainy and snowy road conditions. It is used to take the non-linear operations related to the shape or morphology of features in an image. In the project, morphologyEX as a library import from OpenCV is used to conduct the noise removal process. Two morphological techniques including opening and closing are conducted sequentially to remove the noises. While opening techniques are used to removal the small noises around the objects (snow flake, rain drops) in the background, closing technique removing the noises overlapping the objects in the foreground. Both techniques are used to minimize the noises in rainy and snowy weather. 
 
5.2. Algorithms 
As previous report mentioned that there are five procedures to complete this project, as data pre-processing, model selection, training, testing and improvement. Each procedures have many alternatives, and the team will explain more about neural network model in the following report. The neural network model is the core component to complete adverse conditional object detection task, due to no matter how we pre-process the image, different training epoch, the neural model defines the project height. There are various models available, like Mask-RCNN, fast-RCNN, YOLO, SSD. Therefore, we did some preliminary research and decided to aim YOLO and SSD. There are three models that we will explain, YOLO V5, YOLO V3 and MobileNet. 
5.2.1. YoloV5 
The first alternative to training the model is YOLOV5 which is an object recognition and local algorithm based on neural network. In the input terminal YOLOV5 use the Mosaic augmentation to do the data enhancement, it used 4 pictures to do the random scaling, slipping and random layout. The reason use Mosaic augmentation is in our project’s data set the small object and large object do not have an even proportion, to rich the data set and balance the different type of the detection of the data set (especially the random scaling adds smaller object). Small objects can also be detected in extreme weather conditions. In the Yolo algorithm for the different data set it will set an anchor frame with initial length and width. And different with YoloV3, YoloV5 add this function to the code which means it can optimize the anchor frame value in different training sets adaptively in each training.  
 
Figure 1. Network Architecture of YoloV5 [1] 
As shows in Figure 1 the YoloV5 consists of 3 parts: backbone, Neck and Head. The data first input to CSPDarkent to do the feature extraction and fed to PANet for feature fusion, finally YOLO layer outputs detection results [1]. In the backbone the focus structure could use the slice operation, use at least 32 convolutional kernel extract features from the data set.  
The YoloV5 has less training time, more flexible and faster detection time compared to the other yolo training algorithm. 
5.2.2. YoloV3 
As YOLOV3 has the similar base features as convolutional neural work with YOLOV5, it applied a single neural network to a whole image to make object detection. However, the neural network is constructed by a total of 106 layers with 53 of them are convolutional layers. Each convolutionally layer is followed by a batch normalization layer and a leaky ReLU activation function. There are no pooling layers in YOLO V3 but additional convolutional layers are added to down sample the feature maps to prevent loss of low-level features. Therefore, YOLOV3 improve the ability to detect small objects compare to the other YOLO model. The neural network uses 3 layers at layer 82, 94 and 106 to down sample the input image by factors of 32 ,16 and 8 separately. By applying different network strides to the input image, different sizes of the objects are detected. (32 for small objects, 16 for medium objects, 8 for large objects). While YOLO divides the image into grid cells and make prediction based on each cell’s probability. YOLOV3 apply 1x1 detection kernels to each grid of cells and 1 cell is responsible for detecting one object. In a nutshell, YOLOV3 has reliable accuracy in detecting the small objects in trade of processing speed as more convolutional layers is added to down sample the features. 
 
Figure2. YOLOV3 Convolutional Neural Network Architecture [2] 

5.2.3.MobileNet 
 
Figure 3. MobileNet V2 Convolutional layer Architecture [3] 
As the Figure 3 shows, the bottleneck block is the foundation of the MobileNet V2 and it has three layers. The first layer known as expansion layer which has 1x1 convolution and the purpose is to expand the number of channels by the expansion factor, thus, it will output more channels than the inputs. Then, the depthwise layer has 3x3 convolution and it does not have pooling layers, but sometimes it has a stride of 2 to reduce the spatial dimensions. In addition, the batch normalization will follow the depthwise layer and it will end with the activation function “ReLU6”. Moving into the final layer named as projection layer, the channels will reduce in this layer. For example, the input data has 24 channels, it will expand to 144 channels in expansion layer with expansion factor as 6. Then it will reduce to 24 channels again in projection layer to reduce the data flow through the network. The advantage is to save computation time as it reduces the channels within the network. 
As the Figure 4 shows, it is overall MobileNet V2 architecture. The MobileNet network will convert the input images into features that can describe the image contents and it will pass to the following convolution layer.  
 
Figure 4. Block chain of the MobileNet-SSD [3]

To compete the performances of three different object detection neural networks (YOLOv3, YOLOv5, MobileNet-SSD) mentioned above. The test result from International Research Journal of Engineering and Technology is attached to measure the precision of the model and the performance in the processing units at different levels. (Tesla T4 as high-Tier, 1660 Ti as mid-Tier, Jetson Nano as low-Tier As we can see from the results from Table 1, although YOLOv3 has a relatively high precision in making object classification, the frames per second of the resultant video is lower than the other two models in all three levels of processing units. As we used the model to make a real-time object detection on the road in severe weather, the processing speed of the model must be capable of making emergency input of the passing traffic and pedestrians at once. In trade of the precision, we set YOLO V5 with high FPS in all ends of processing unit as our prior choice of implementing of our object detection model. While the model may have lower prediction accuracy than YOLOv3, it should be reliable in detecting the large-size and mid-size objects. In our case, the mosaic augmentation of YOLOv5 is significantly benefit for fluent function of the model in the severe weather in the case that the vision is ambiguous. 
Model 	Mean Average Precision (%) 	FPS on Tesla T4 	FPS on 1660 Ti 	FPS on Jetson Nano 
YOLOv3 	54.3 	80 	21 	8 
YOLOv5 	37.6 	100 	28 	15 
MobileNet-SSD 	33.7 	94 	26 	15 
Table 1. Performance Analysis on Three Neural Network Models
5.3. Dataset
The CADCD dataset has 64,400 labelled images with the timestamp and LiDAR point cloud, and it occupies 91.3 GB storage. The image is 1280x1024 pixels and depth is 24. To reduce the training time, the image could be reduced to a smaller pixel’s combination. The team needs to define the probability of pixel reduction. As the dataset has labelled and annotations which the dataset has pre-processed and ready to be used in training. The last step is to split the data as 70% in training set, 25% in validation set and 5% in testing set.
5.4. Training	
While training the YOLOv5 model, a set of training parameters is defined according to the uniqueness of the datasets. The image size is set to 640 to make sure there is enough details in the image for the model to learn. The batch size is set to 16 and epochs is set to 400 as we only have a very limited number of images in the dataset and we want the model to learn as much as possible in the dataset. The learning rate is set from 0.01 at initial time and increased to 0.1 at the final time. The optimizer is set to the default SGD to update the weights in more iterations.
  

5.5. Hardware
There are two alternatives, one is to use google Colab platform, another is to use our desktop. Because we are three members group, the GPUs for us are Nvidia GTX970, Nvidia GTX1070 and Nvidia GTX1070 Ti. So, regarding the group’s GPUs are not high-performance hardware, so the google Colab is a better platform to provide stable and speed for this project. However, the google Colab has RAM and Disk limitation so it might need to reduce the dataset. In this case, the less images may affect the accuracy, but it will significantly save the training and testing time.  
6. Experimental Evaluation

7. Results
7.1. Attentive Generative Adversarial Network
               
7.2. Image Lightness
 
 

7.3. Yolov5:
 
Regarding to the architecture of YOLOv5 model applied in the project, it has 24 layers of neurons. The layer 0 to layer 9 is the backbone structure of the layer. It basically is a separate convolutional neural network used to extract the feature maps of the important features from the training images. The model neck is from layer 10 to layer 23 which is used to generate feature pyramids. It helps to identify the objects in different sizes images using the multiple convolution and pooling layers. As the feature pyramids classify the feature maps from the top to the bottom ones, the neural network should be able to detect different objects at different scales in various sizes of images. Finally, the 24th layer is the head part of the neural network used to perform the final detection. It is going to draw the bounding boxes and output the classification result according to the anchor boxes generated on the features.
Makesense is an online annotation tool which can be used to output the annotation file in YOLO format. By inputting the training images after the preprocessing steps (noise removal and the brightness adjustment), four classes of objects in the images are labelled. (car, person, traffic light and buildings) There is 109 images in rainy, snow and cloudy weathers are used for training and 20% of the images are used for validation. The annotation file is output in txt format and there is 109 annotation files in total corresponding to 109 images. The class information and the bounding box is recorded in the annotation files.

 
As we can see from the validation result, the model is able to detect almost every labelled object in the images. As there is over 400 samples of cars in the 109 images, the model is capable of detecting and classifying the cars at a relative high accuracy around 80%. However, as there is not many samples of person in the training data, the model is not confident enough to determine the person in the images. Also, the model is strong at detecting the closed objects which is ideal for the live video detection on the road.
 
 







 

8. Appendix (Code)

