# Problem Set 3: Geometry

All program outputs are logged; a copy is placed in the `output` directory of this directory. Additionally, all images generated are copied to the output directory.

## Problem 1: Calibration

### Part a
Matrix M recovered from normalized points using least squares:
```
[ 0.76786      -0.49385     -0.023398    0.0067443    
  -0.085213    -0.091468    -0.90652     -0.087757    
  0.18265      0.29883      -0.074192    1         ]

Projected 3D point
[1.2323, 1.4421, 0.4506, 1]
to 2D point
[0.14190573, -0.45183986, 0.99999994]
Residual = 0.00156357
```

Matrix M recovered from normalized points using singular value decomposition:
```
[ 0.45828      -0.29474     -0.013957    0.0040259    
  -0.050856    -0.054585    -0.54106     -0.052376    
  0.10901      0.17835      -0.044268    0.59682   ]

Projected 3D point
[1.2323, 1.4421, 0.4506, 1]
to 2D point
[0.1419062, -0.45184308, 1]
Residual = 0.0015622
```

### Part b
The sets of 8, 12, and 16 points for this problem are pseudorandomly selected as determined by the seed in the `config/ps3.yaml` configuration file (found in the top level of this repository). This seed is passed to a Mersenne Twister Engine. The following residuals and matrices were generated with random seed `16 38 c7 e4 6a a2 d8 cc 96 f6 fe f1 4b 7d a7 25`.

Average residuals for each trial of k:
```
     8 points           12 points          16 points
[4.946314677278304, 10.93386707473356, 3.778666987456681;
 14.73268176079164, 307.9845094072188, 45.5595627642943;
 23.78386246786605, 12.2410688372804, 10.69169865005561;
 42.80109819671209, 7.643525226174578, 7.02850997446868;
 574.9949718217316, 18.77454368333919, 7.20265203350789;
 23.45316473037575, 117.4753877977528, 31.98030631790589;
 1764.827066954847, 16.46992913351448, 165.5090534473709;
 8.496700939337815, 15.42366869637113, 22.60050929982096;
 2.451639696679493, 15.77386195904591, 6.830741396911162;
 20.53951516110012, 5.760961837178528, 5.929776959269682]
```

Best matrix M:
```
Minimum residual: 2.45164
Found with constraint size: 8
Computed parameters:
[ -2.098       1.3374       0.46848      210.25       
  -0.4441      -0.30386     2.3166       157.21       
  -0.0022486   -0.0011002   0.00061062   1      ]
```

### Part c
Center of the camera in the world's 3D coordinates:
```
[302.75244;
 307.05109;
 30.453285]
```

## Problem 2: Fundamental Matrix Estimation

### Part a
Least squares estimate of the fundamental matrix F:
```
[ -6.5904e-07  7.8708e-06   -0.0018749   
  8.8167e-06   1.2364e-06   0.01716      
  -0.00091146  -0.026341    1         ]
```

### Part b
Fundamental matrix F after reduction from rank 3 to rank 2
```
[ -5.3273e-07  7.864e-06    -0.0018749   
  8.8282e-06   1.2357e-06   0.01716      
  -0.00091147  -0.026341    1         ]
```

### Part c
Pic A and Pic B with epipolar lines drawn on them:

<img src="./output/ps3-2-c-1.png" width="350"/>    <img src="./output/ps3-2-c-2.png" width="350"/>

## Problem 2 Extra Credit: Normalization

### Part d
Transform matrix for the points from Pic A, T_a:
```
[ 0.0010604    0            -0.59274     
  0            0.0010604    -0.34528     
  0            0            1        ]
```

Transform matrix for the points from Pic B, T_b:
```
[ 0.00093985   0            -0.57961     
  0            0.00093985   -0.32603     
  0            0            1        ]
```

Fundamental matrix F_hat:
```
[ 6.9756       -96.948      -3.4738      
  -60.857      18.526       -233.36      
  -15.566      194.67       0.84348  ]
```

### Part e

A better fundamental matrix F:
```
[ 6.9523e-06   -9.6624e-05  0.02431      
  -6.0654e-05  1.8464e-05   -0.19143     
  0.00024596   0.25962      -5.7308  ]
```

Pic A and Pic B with epipolar lines drawn on them, using the normalized fundamental matrix above:

<img src="./output/ps3-2-e-1.png" width="350"/>    <img src="./output/ps3-2-e-2.png" width="350"/>