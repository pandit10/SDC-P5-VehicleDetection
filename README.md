# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png

[image8]: ./md_images/test1.jpg
[image9]: ./md_images/test2.jpg
[image10]: ./md_images/test3.jpg
[image11]: ./md_images/test4.jpg
[image12]: ./md_images/test5.jpg
[image13]: ./md_images/test6.jpg

[image14]: ./md_images/full_pipeline/test1.png
[image15]: ./md_images/full_pipeline/test2.png
[image16]: ./md_images/full_pipeline/test3.png
[image17]: ./md_images/full_pipeline/test4.png
[image18]: ./md_images/full_pipeline/test5.png
[image19]: ./md_images/full_pipeline/test6.png

[image7]: ./examples/output_bboxes.png



[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 49 through 60 of the file called `classifier.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and judged performance based on the human ability to differentiate vehicles from non-vehicles by simply looking at the HOG image representation. After collecting various combinations of parameters, I moved to the next step which was training a classifier to differentiate vehicles from non-vehicles based on the extracted features, and further narrowing down the parameter choices based on the classifier test accuracy.
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in the file `classifier.py` specifically in lines 82 through 87, using different parameter combinations. Finally I settled for the following parameters based on classifier test/validation accuracy:

|Parameter |--------->| value|
|----------||--------|
|color space||'YCrCb'|
|HOG orientations||9|
|HOG pixels per cell||8|
|HOG cells per block||2|
|HOG channel||"ALL"|
|Spatial binning dimensions||(16, 16)|
|Number of histogram bins||16|
|Spatial features on or off||True|
|Histogram features on or off||True|
|HOG features on or off||True|



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used three window sizes to search the frame for classified vehicles, depending on the location in the image. Windows with 128 px size are used to detect new vehicles entering the frame from the sides, 96 px windows were used to detect vehicles in the near range across the image, as well as 64 px windows to detect far range vehicles. Over lapping values where mainly chosen by experimenting which values yielded better results.



#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a robust result.  Here are some example images:

![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_cars.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In the lines 171 through 193 in the file `search_classify.py`, I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames with detected windows, their corresponding heatmaps, and the output from the thresholded heatmap labels:

![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

##### Challenges
* The hardest challenge was finding the most efficient feature vector to train the classifier and use to predict image windows. In addition, finding the best parameters for the HOG feature was very challenge in the sense of finding a balance point between prediction accuracy and number of features.
* Due to the slow processing of the video, testing was very time consuming. Any way of visualizing the output during runtime consumed even more time.
* False positives are handled by the heat map, however, false negatives represent a huge weakness for this system, especially regarding vehicles in the far range.

##### Improvements and Future work
* A complex tracking algorithm should be implemented to improve the quality of the detections. A kalman filter would be a great approach. When we first predict a vehicle, we initialize an estimate, we then predict the location and dimensions for the created estimates in the next frame. When we detect new observations, we associate every observation to one of the previous estimate, and make use of our previous knowledge to smoothen the fluctuations in position and dimensions of the detections.
* Using perspective transform to create variable sized search windows. This will improve classifier predictions by improving window coverage for the whole vehicle.
