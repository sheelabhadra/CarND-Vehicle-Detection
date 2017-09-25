
## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/sample_image.jpg
[image2]: ./examples/sample_hog.jpg
[image3]: ./examples/test_image_results.jpg
[image4]: ./examples/heat_map.jpg

---
### Writeup

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.  

I extracted the HOG features from images using the `hog()` method from the `skimage.feature` object. More information on the `hog()` method can be found [here](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html). The code for the HOG feature extraction is contained in the fourth code cell of the IPython notebook.  

The feature extraction was performed on the dataset of cars and non-cars. The `vehicles` dataset was collected from https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip and the `non-vehicles` dataset was collected from https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Finally i settled with the `YCrCb` color space as it gave the best test accuracy with the SVM classifier.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8,8)` and `cells_per_block=(2,2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of `orientation`, `pixels per cell`, `cells per block` with various combination of color channels and found that the combination of using all the 3 channels of `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8,8)` and `cells_per_block=(2,2)` gave the best test accuracy with the SVM classifier.  

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Apart from HOG features I also used spatial binning features and the histogram features to train a linear SVM as all these features gave high accuracy. After some experimentation histogram bin size of 32 and spatial bin size of 32x32 seem to give the best test accuracy. I consistently got a test accuracy in the range of 98.5% and 99.5% by incorporating all the features for training the linear SVM classifier.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented the sliding window search by searching in the region in the y direction between 400 and 656 pixels as it is the most likely region where the vehicles can be found. I searched the region with scale values of [1, 1.25, 1.5, 1.75, 2.0]. With scale values around 1, there were a few false positives while a few false negatives were generated when the scale factor was 2. With a scale factor of 1.5 the sliding window search performed well. I used a window overlap of 50% between adjacent windows and it worked well for me. 


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/watch?v=Sv2S-TIJFv0&feature=youtu.be) on YouTube.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The positions of positive detections in each frame of the video are found using the `find_cars()` method.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  Thereafter, I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a few test frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text][image4]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I faced a few issues while deciding the scaling factor suitable for the pipeline. Ultimately I had to settle for a scaling factor that was a tradeoff between false positives and accuracy.  

Although my pipeline performs decently when the weather conditions are good, it might suffer in bad weather conditions. It can be also observed in my video that at times my pipeline detects cars pretty late after entering the video frame.

HOG+SVM is a simple object detection algorithm and not a dedicated object tracking algorithm per se. There are more robust deep learning based algorithms such as `YOLO` and `SSD` that are known to perform very well in real-time object detection taska. The classifier in these algorithms do not rely on handmade features such as HOG but rather learn relevant features on their own with the help of Convolutional Neural Networks. Apart from these a lot of state-of-the-art object tracking algorithms are listed on [the KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/eval_tracking.php). I would definitely love to continue working on the project and explore these algorithms to make a more robust vehicle tracking system.


```python

```
