# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3a]: ./output_images/sliding-windows/test1.jpg
[image3b]: ./output_images/sliding-windows/test2.jpg
[image3c]: ./output_images/sliding-windows/test3.jpg
[image3d]: ./output_images/sliding-windows/test4.jpg
[image3e]: ./output_images/sliding-windows/test5.jpg
[image3f]: ./output_images/sliding-windows/test6.jpg
[image4a]: ./output_images/candidate-windows/test1.jpg
[image4b]: ./output_images/candidate-windows/test2.jpg
[image4c]: ./output_images/candidate-windows/test3.jpg
[image4d]: ./output_images/candidate-windows/test4.jpg
[image4e]: ./output_images/candidate-windows/test5.jpg
[image4f]: ./output_images/candidate-windows/test6.jpg
[image5a]: ./output_images/heat-map/test1.jpg
[image5b]: ./output_images/heat-map/test2.jpg
[image5c]: ./output_images/heat-map/test3.jpg
[image5d]: ./output_images/heat-map/test4.jpg
[image5e]: ./output_images/heat-map/test5.jpg
[image5f]: ./output_images/heat-map/test6.jpg
[image6a]: ./output_images/detected-cars/test1.jpg
[image6b]: ./output_images/detected-cars/test2.jpg
[image6c]: ./output_images/detected-cars/test3.jpg
[image6d]: ./output_images/detected-cars/test4.jpg
[image6e]: ./output_images/detected-cars/test5.jpg
[image6f]: ./output_images/detected-cars/test6.jpg
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png
[video1]: ./output_videos/project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

```python
### Tweak these parameters and see how the results change.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used the extract_features methdo from the lessons as is. Since the image dataset was small, I decided to use all the three types of features (HOG, histogram and spatial). I also decided to use the YCrCb colorspace because it was giving the best results. For spatial and histogram features I used the default parammeters because they were already tweaked in the lessons.
For HOG I chose ALL hog channels with 9 orientations. This is because selecting a single hog channel was not working for the classifier. I used a Linear SVM which is known to work well for HOG features. I scaled the features using StandardScaler and split 20% of data randomly into a test set. The statistics of he whole process are printed below:

    38.0 seconds to extract car features...
    38.13 seconds to extract non-car features...
    4.88 seconds to scale/transform features...
    Using: 9 orientations 8 pixels per cell and 2 cells per block
    Feature vector length: 8460
    8.32 seconds to train Linear SVM...
    Test Accuracy of SVC =  0.9879

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3a]
![alt text][image3b]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:
![png](output_10_0.png)

![alt text][image6a]
![alt text][image6b]
![alt text][image6c]
![alt text][image6d]
![alt text][image6e]
![alt text][image6f]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](../output_videos/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image4a]
![alt text][image4b]
![alt text][image4c]
![alt text][image4d]
![alt text][image4e]
![alt text][image4f]
---
![alt text][image5a]
![alt text][image5b]
![alt text][image5c]
![alt text][image5d]
![alt text][image5e]
![alt text][image5f]
---


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image5f]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image6f]


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The great thing about this project was that most of the code was already provided and explained in the lessons. Most of the code works as it is and the only challenging part is tuning the various parameters.
The only change I made is in the find_cars function where instead of drawing rectangles on the image I am returning the list of bounding boxes similar to search_windows method so that we can add heat and apply thresholds to these bounding boxes as well.

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

