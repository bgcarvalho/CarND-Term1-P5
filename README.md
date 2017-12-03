# Vehicle detection and Tracking

## Project Report

This project consists of using computer vision to detect vehicles in images
and track vehicles' positions in a video feed.

## Extrating image HOG features

To extract image HOG features it is used `skimage.feature.hog` function.
The main arguments are:

- Input image with only one color channel
- orientations = 9: different orientations to look for from completly vertical to 
completly horizontal

And:
- pixels_per_cell = 8
- cells_per_block = 2

The image is divided in windows, each window in cells and each cell in N x N 
pixels. Also, a group of cells is a block 
(More info)[http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html].

The color space chosen was YCrCb, and it was decided to use HOG features to
all channels, as in lines from `extract_train_features` function:

```python

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

The function `np.ravel` does a reshpae in the vetor, turning it into 1-D array.

These features were fed into a Support Vector Machine Classifier (SVC) after
normalization. The normalization was conducted using 
`from sklearn.preprocessing.StandardScaler` in lines (note the `float64` type):

```python

X = np.concatenate((car_features, notcar_features), axis=0).astype(np.float64)  
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)

```









Raw colors (`bin_spatial`) and histogram (`color_hist`) are also reshaped to
1-D and then all three concatenated to form the feature vector.

The `cells_per_step = 2` variable controlls the overlapping in search, 
in this case equals to 75% (Ref)[https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/fd66c083-4ccb-4fe3-bda1-c29db76f50a0/concepts/c3e815c7-1794-4854-8842-5d7b96276642].


## Discussions
- LUV color space can produce infinity values, breaking the pipeline
- YCrCB color space works fine
- A bug in [0 - 1] <=> [0 255] intervals convertions took me a few days to solve
- Processing the search every 8 or 10 frames helps to speed up the algorithm,
but is only enough to process 3 frames/s

## References
