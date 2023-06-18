# Detection and Segmentation

- Classification: assign a label or a category to the entire image or a specific region within it;
- Semantic segmentation: assign semantic values to each pixel within the image;
- Object detection: identify, classify (assign a label) and localize (find a box that contains the object) multiple objects within the image;
- Instance Segmentation: combines object detection and semantic segmentation.

# Semantic segmentation

Involves assigning a semantic label to each pixel in an image, thereby dividing the image into multiple segments based on their categories. Unlike object detection, which focuses on localizing and identifying specific objects, semantic segmentation provides a more fine-grained understanding of the image by labeling each pixel. 

The idea of using downsampling and upsampling is a common approach in semantic segmentation.
The process typically involves two main steps: the encoder (downsampler) and the decoder (upsampler):

- The downsampling phase involves progressively reducing the spatial dimensions of the input image while increasing the number of feature channels. This is achieved by applying convolutional layers. As the network descends deeper into the encoder, the receptive field increases, allowing for the extraction of high-level abstract features. Downsampling helps capture global context and large-scale structures in the image.

- The upsampling phase aims to recover the spatial resolution of the feature map while retaining the learned features. Upsampling can be achieved through different techniques, including unpooling, strided (stride > 1) transposed convolutions (stride is the ratio between kernel movement in output and input), or interpolation.

# Object detection

The goal is to detect the presence of objects, determine their classes or categories, and provide bounding box coordinates that enclose each object instance.

Region-based Convolutional Neural Networks (R-CNNs) are a class of deep learning models that have significantly advanced object detection performance.

The R-CNN family of models consists of several iterations, including R-CNN, Fast R-CNN, and Faster R-CNN:

- R-CNN (Region-CNN): The original R-CNN approach operates in a two-step process. First, it generates a set of region proposals using selective search or a similar algorithm. Then, each region proposal is independently classified using a convolutional neural network. Finally, bounding box regression is applied to refine the region proposals' positions. 
    
    Problem: say a Region of Interest method proposes ~2k regions, we need to do ~2k forward passes for each image.

    Solution: we can pass the image through a CNN before cropping and crop the features instead.

- Fast R-CNN: Fast R-CNN addresses some of the limitations of R-CNN by introducing a shared feature extraction network. Instead of processing each region proposal independently, the entire image is fed through a convolutional network (like a CNN) to extract a feature map. Region of Interest (RoI) pooling is then applied to align the region proposals with the extracted features. These aligned features are used to classify objects and refine their bounding box positions.

    Problem: runtime dominated by region proposal algorithm. 


- Faster R-CNN: Faster R-CNN further improves the efficiency of object detection by introducing a region proposal network (RPN) as part of the architecture. The RPN operates on the shared convolutional feature map to generate region proposals directly, eliminating the need for external proposal algorithms. The RPN suggests potential regions of interest, which are then fed into the Fast R-CNN network for classification and bounding box regression.

However, R-CNNs have some drawbacks. They are computationally expensive, as they involve multiple forward passes through the CNN for each region proposal. This limits their real-time applicability. Additionally, the two-stage process of R-CNNs can be time-consuming during inference.

To address these limitations, newer approaches like Single Shot MultiBox Detector (SSD) and You Only Look Once (YOLO) have emerged. These methods aim to achieve real-time object detection by directly predicting object classes and bounding box coordinates from a single pass of the neural network.

# Instance segmentation

Instance segmentation is a computer vision task that combines object detection and semantic segmentation to identify and delineate individual objects within an image. While semantic segmentation classifies every pixel in an image into predefined classes, instance segmentation goes a step further by assigning a unique label to each distinct instance of an object.

One popular instance segmentation model is Mask R-CNN (Region-based Convolutional Neural Networks with Masking). Mask R-CNN is an extension of the Faster R-CNN object detection model. It adds a branch to the model that generates a binary mask for each detected object instance, along with the class label and bounding box coordinates.

# U-Net

U-Net is a popular convolutional neural network (CNN) architecture designed for image segmentation tasks. 

Its main components are:

- Encoder: The encoder part of the U-Net consists of a series of convolutional and pooling layers. These layers progressively downsample the input image, extracting high-level features and capturing contextual information. The encoder acts as a feature extractor, gradually reducing the spatial dimensions while increasing the number of channels.

- Bottleneck/Bridge: The bottleneck or bridge connects the encoder and decoder parts of the network. It typically consists of a single convolutional layer, which serves as a transition from the contracting path (encoder) to the expanding path (decoder). This bridge helps in preserving spatial information during the transition.

- Decoder: The decoder part of the U-Net is responsible for upsampling the feature maps and generating the final segmentation output. It consists of a series of upsampling and concatenation operations, along with convolutional layers. The upsampling layers increase the spatial dimensions, while concatenation merges the feature maps from the corresponding encoder layers to provide both local and global information for accurate segmentation. The decoder progressively recovers the spatial resolution and refines the segmentation output.

