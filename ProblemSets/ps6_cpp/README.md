# Problem Set 6: Particle Tracking

## Problem 1: Particle Filter Tracking

### Part a
Particle filter using mean squared error as a measure of similarity.

Frame 28
<br>
<img src="./output/ps6-1-a-f28.png" width="640"/>

Frame 84 
<br>
<img src="./output/ps6-1-a-f84.png" width="640"/>

Frame 144
<br>
<img src="./output/ps6-1-a-f144.png" width="640"/>

### Part e
Particle filter being run on a noisy video stream.

Frame 14
<br>
<img src="./output/ps6-1-e-f14.png" width="640"/>

Frame 32
<br>
<img src="./output/ps6-1-e-f32.png" width="640"/>

Frame 46
<br>
<img src="./output/ps6-1-e-f46.png" width="640"/>

## Problem 2: Appearance Model Update

### Part a
The same particle filter as before, but the appearance model is updated between frames to be a blended version of the previous model and the current best estimate of the tracked object's position.

Frame 15
<br>
<img src="./output/ps6-2-a-f15.png" width="640"/>

Frame 50
<br>
<img src="./output/ps6-2-a-f50.png" width="640"/>

Frame 150
<br>
<img src="./output/ps6-2-a-f150.png" width="640"/>

### Part b
The same particle filter as in part a, but being run on a noisy video stream.

Frame 15
<br>
<img src="./output/ps6-2-b-f15.png" width="640"/>

Frame 50
<br>
<img src="./output/ps6-2-b-f50.png" width="640"/>

Frame 150
<br>
<img src="./output/ps6-2-b-f150.png" width="640"/>

## Problem 3: Mean Shift Lite

### Part a
Using a histogram with 32 bins as the appearance model instead of a copy of the image patch. Similarity is measured by chi-squared error.

Frame 28
<br>
<img src="./output/ps6-3-a-f28.png" width="640"/>

Frame 84 
<br>
<img src="./output/ps6-3-a-f84.png" width="640"/>

Frame 144
<br>
<img src="./output/ps6-3-a-f144.png" width="640"/>

### Part b
Attempting to track Romney's hand using the histogram as the appearance model. This did not work.

Frame 15
<br>
<img src="./output/ps6-3-b-f15.png" width="640"/>

Frame 50 
<br>
<img src="./output/ps6-3-b-f50.png" width="640"/>

Frame 140
<br>
<img src="./output/ps6-3-b-f140.png" width="640"/>