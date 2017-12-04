# Vehicle detection and Tracking

## Project Report

This project consists of using computer vision to detect vehicles in images
and track vehicles' positions in a video feed.

- Output video is [here](./project_video_overlay.mp4)
- jupyter notebook is [here](./Notebook.ipynb)

[//]: # (images)
[data_set]: ./output_images/data_set_examples2.png
[hog]: ./output_images/hog_examples.png
[features]: ./output_images/feature_vectors.png
[test]: ./output_images/test_model.png
[searchs1]: ./output_images/searchwin_s1.png
[heatmap]: ./output_images/heatmap.png
[video1]: ./output_images/video_out1.png


## Extrating image HOG features

At first, as a good practice we should look some examples in the dataset,
cars and not cars.

![Cars and Not Cars dataset examples][data_set]

To extract image HOG features it is used `skimage.feature.hog` function.
The main arguments are:

- Input image with only one color channel
- orientations = 9: different orientations to look for, varying from 
completly vertical to completly horizontal.

And:
- pixels_per_cell = 8
- cells_per_block = 2

The image is divided in windows, each window in cells and each cell in N x N 
pixels. Also, a group of cells is a block 
(More info)[http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html].

After trying different combinations of values, the above are those which
produced good results.

The color space chosen was YCrCb, and it was decided to use HOG features to
all channels, as in lines from `extract_train_features` function:

```python

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image

for channel in range(feature_image.shape[2]):
    imgrs = feature_image[:, :, channel].copy()
    hog_features.append(get_hog_features(imgrs, 
                                         orient, 
                                         pix_per_cell, 
                                         cell_per_block, 
                                         vis=False, 
                                         feature_vec=True))
hog_features = np.ravel(hog_features)

```

The function `np.ravel` reshapes the array turning it into 1-D vector.


![HOG examples for some images][hog]

These features were fed into a Support Vector Machine classifier (SVC) after
normalization (zero mean, unit variance). The normalization was conducted using 
`sklearn.preprocessing.StandardScaler` in lines (note the `float64` type):

```python

X = np.concatenate((car_features, notcar_features), axis=0).astype(np.float64)  
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

```

The scikit-learn scaler works best with with `np.float64` data type.


### Sliding search

The model was trained for 64x64 car images. In order to properly search cars 
in the camera image it is used a sliding window tecnique. The algorithm walks
through the image asking the SVC to predict smaller regions. For each of these
windows it is calculated the feature vector with raw color, histogram and HOG.

![search windows for scale = 1][searchs1]

The `cells_per_step = 2` variable controlls the overlapping in search, 
in this case equals to 75% [Class](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/fd66c083-4ccb-4fe3-bda1-c29db76f50a0/concepts/c3e815c7-1794-4854-8842-5d7b96276642).

Together with HOG features are also use raw colors (`bin_spatial`) and 
histogram (`color_hist`), both reshaped to
1-D and then all three concatenated. This feature vector is chosen to have
enough "car information" and reduce false positives.

![Feature vectors plotted][features]

Also a heat map is used to increase confidence by suming detections of the 
same car and then applying a threshold (note that the leftmost detection is
discarted).

![Heatmap reduces false positives][heatmap]


## Video

After tunning the pipeline, functions were joined in a Python class and enable
more control over processing sequential images (video frames).

Implementation is [here](./car_search.py), and it uses lane detection from the
previous project [copied here](./lane_search.py).

![video output][video1]

Full video is [here](./project_video_overlay.mp4).


## Final notes
- LUV color space can produce infinity values, breaking the pipeline
- YCrCB color space works fine
- A bug in [0 - 1] <=> [0 255] intervals convertions took me a few days to solve
- Processing the search every 8 or 10 frames helps to speed up the algorithm,
but is not enough to consider the algorithm fast. For example, it couldn't 
process video in real time (24 FPS)
- It is possibile to reduce regions of search, use less scales
- It could be possible to find a even better color filter combination
- Time to extract features before training: 100s
- Time to train: 27s
- Time to predict cars in a single frame: 0.48s (testing with four scales)

