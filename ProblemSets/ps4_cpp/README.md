# Problem Set 4: Harris, SIFT, and RANSAC

All program outputs are logged; a copy is placed in the `output` directory of this directory. Additionally, all images generated are copied to the `output` directory.

## Problem 1: Harris Corners

### Part a
Computing X and Y gradients for transA/B and simA/B images.

transA and transB:
<br>
<img src="./output/ps4-1-a-1.png" width="700"/>
<br>

simA and simB:
<br>
<img src="./output/ps4-1-a-2.png" width="700"/>
<br>


### Part b
Harris values (corner responses) for the transA/B and simA/B image pairs. Very dark sections correspond to areas with one high and one low eigenvalue, i.e. they indicate lines. Medium grey areas are flat regions; whiter areas have strong corner responses.

transA | transB
--- | ---
<img src="./output/ps4-1-b-1.png" width="350"/> | <img src="./output/ps4-1-b-2.png" width="350"/>

simA | simB
--- | ---
<img src="./output/ps4-1-b-3.png" width="350"/> | <img src="./output/ps4-1-b-4.png" width="350"/>

### Part c
Corner points found in the image after thresholding and non-maximum suppression.

transA | transB
--- | ---
<img src="./output/ps4-1-c-1.png" width="350"/> | <img src="./output/ps4-1-c-2.png" width="350"/>

simA | simB
--- | ---
<img src="./output/ps4-1-c-3.png" width="350"/> | <img src="./output/ps4-1-c-4.png" width="350"/>

## Problem 2: SIFT Features

### Part a
Angles computed from the gradients of each image at each corner point.

transA and transB:
<br>
<img src="./output/ps4-2-a-1.png" width="700"/>
<br>

simA and simB:
<br>
<img src="./output/ps4-2-a-2.png" width="700"/>
<br>

### Part b
SIFT descriptors are computed matched between images. Putative matches are drawn on the below image pairs.

transA and transB:
<br>
<img src="./output/ps4-2-b-1.png" width="700"/>
<br>

simA and simB:
<br>
<img src="./output/ps4-2-b-2.png" width="700"/>
<br>

## Problem 3: RANSAC

In all of these problems, "Best transform" refers to the transformation from the "A" image to the "B" image.

### Part a
RANSAC is used to find the best translation transformation in the transA/B image pair. The consensus set for this translation is drawn on the images.

<img src="./output/ps4-3-a-1.png" width="700"/>

```
Best transform =
[1, 0, -134;
 0, 1, -78]
with consensus ratio 0.213675
```

### Part b
The same process as above was done to compute the similarity transformation in the simA/B image pair.

<img src="./output/ps4-3-b-1.png" width="700"/>

```
Best transform =
[0.97133934, -0.28252175, 38.384468;
 0.28252175, 0.97133934, -58.787109]
with consensus ratio 0.628205
```

## Extra credit problems
A continuation of problem 3.

### Part c
Assuming instead that the transformation between simA and simB is affine (instead of similar):

<img src="./output/ps4-3-c-1.png" width="700"/>

```
Best transform =
[0.95664239, -0.26735318, 39.611755;
 0.28754473, 0.99733996, -65.212708]
with consensus ratio 0.602564
```

### Part d
Warping simB back to simA is done by computing the affine warp of the image using the inverse of the **similarity** transformation computed in part b.

<img src="./output/ps4-3-d-1.png" width="350"/>

### Part e
Warping simB back to simA is done by computing the affine warp of the image using the inverse of the **affine** transformation computed in part b.

<img src="./output/ps4-3-e-1.png" width="350"/>