# Problem Set 5: Optic Flow

## Problem 1: Lucas-Kanade Optical Flow

### Part a
X and Y displacements from `Shift0` to `ShiftR2`

<img src="./output/ps5-1-a-1.png" width="300"/>

X | Y
--- | ---
<img src="./output/ps5-1-a-1-uColorMap.png" width="300"/> | <img src="./output/ps5-1-a-1-vColorMap.png" width="300"/>

From `Shift0` to `ShiftR5U5`

<img src="./output/ps5-1-a-2.png" width="300"/>

X | Y
--- | ---
<img src="./output/ps5-1-a-2-uColorMap.png" width="300"/> | <img src="./output/ps5-1-a-2-vColorMap.png" width="300"/>

### Part b
Using the same parameters as in part a:

X and Y displacements from `Shift0` to `ShiftR10`

<img src="./output/ps5-1-b-1.png" width="300"/>

X | Y
--- | ---
<img src="./output/ps5-1-b-1-uColorMap.png" width="300"/> | <img src="./output/ps5-1-b-1-vColorMap.png" width="300"/>

From `Shift0` to `ShiftR20`

<img src="./output/ps5-1-b-2.png" width="300"/>

X | Y
--- | ---
<img src="./output/ps5-1-b-2-uColorMap.png" width="300"/> | <img src="./output/ps5-1-b-2-vColorMap.png" width="300"/>

From `Shift0` to `ShiftR40`

<img src="./output/ps5-1-b-3.png" width="300"/>

X | Y
--- | ---
<img src="./output/ps5-1-b-3-uColorMap.png" width="300"/> | <img src="./output/ps5-1-b-3-vColorMap.png" width="300"/>

## Problem 2: Gaussian and Laplacian Pyramids

### Part a
A Gaussian pyramid with four levels for the Yosemite data set (DataSeq1):

<img src="./output/ps5-2-a-1.png" width="400"/>

### Part b
A Laplacian pyramid of the Yosemite data set

<img src="./output/ps5-2-b-1.png" width="400"/>

## Problem 3: Warping by Flow

#### X and Y displacements for the Yosemite data set.

From image 0 to image 1:

X displacement | Y displacement
--- | ---
<img src="./output/ps5-3-a-1-1-uColorMap.png"/> | <img src="./output/ps5-3-a-1-1-vColorMap.png"/>

Flow vectors | difference
--- | ---
<img src="./output/ps5-3-a-1-1.png"/> | <img src="./output/ps5-3-a-1-1-warped-diff.png"/>

From image 1 to image 2:

X displacement | Y displacement
--- | ---
<img src="./output/ps5-3-a-1-2-uColorMap.png"/> | <img src="./output/ps5-3-a-1-2-vColorMap.png"/>

Flow vectors | difference
--- | ---
<img src="./output/ps5-3-a-1-2.png"/> | <img src="./output/ps5-3-a-1-2-warped-diff.png"/>

#### X and Y displacements for the data set with the dog

From image 0 to image 1:

X displacement | Y displacement
--- | ---
<img src="./output/ps5-3-a-2-1-uColorMap.png"/> | <img src="./output/ps5-3-a-2-1-vColorMap.png"/>

Flow vectors | difference
--- | ---
<img src="./output/ps5-3-a-2-1.png"/> | <img src="./output/ps5-3-a-2-1-warped-diff.png"/>

From image 1 to image 2:

X displacement | Y displacement
--- | ---
<img src="./output/ps5-3-a-2-2-uColorMap.png"/> | <img src="./output/ps5-3-a-2-2-vColorMap.png"/>

Flow vectors | difference
--- | ---
<img src="./output/ps5-3-a-2-2.png"/> | <img src="./output/ps5-3-a-2-2-warped-diff.png"/>

## Problem 4: Hierarchical LK Optic Flow

The problem statement is unclear about how the output should be structured, so here's the flow vectors and displacements (like above) for the Yosemite and Doggo data sets

#### X and Y displacements for the Yosemite data set

From image 0 to image 1:

X displacement | Y displacement
--- | ---
<img src="./output/ps5-4-a-1-uColorMap.png" width="300"/> | <img src="./output/ps5-4-a-1-vColorMap.png" width="300"/>

Flow vectors
<br>
<img src="./output/ps5-4-a-1.png" width="300"/>

From image 1 to image 2:

X displacement | Y displacement
--- | ---
<img src="./output/ps5-4-a-2-uColorMap.png" width="300"/> | <img src="./output/ps5-4-a-2-vColorMap.png" width="300"/>

Flow vectors
<br>
<img src="./output/ps5-4-a-2.png" width="300"/>

#### X and Y displacements for the data set with the dog

From image 0 to image 1:

X displacement | Y displacement
--- | ---
<img src="./output/ps5-4-b-1-uColorMap.png" width="300"/> | <img src="./output/ps5-4-b-1-vColorMap.png" width="300"/>

Flow vectors
<br>
<img src="./output/ps5-4-b-1.png" width="300"/>

From image 1 to image 2:

X displacement | Y displacement
--- | ---
<img src="./output/ps5-4-b-2-uColorMap.png" width="300"/> | <img src="./output/ps5-4-b-2-vColorMap.png" width="300"/>

Flow vectors
<br>
<img src="./output/ps5-4-b-2.png" width="300"/>