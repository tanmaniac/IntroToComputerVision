CS4495 - Spring 2015 - OMS                Introduction to Computer Vision

Problem Set 1: Edges and Lines

Due Sunday, February 8th, 2015 at 11:55pm
=========================================

Description
-----------

This problem set is your first “vision” project where you compute an “answer” – that is some structural or semantic description as to what is in an image. You’ll find edges and objects. And you’ll learn that some methods work well for carefully controlled situations and hardly at all when you relax those constraints.

RULES: You may use Matlab (or image processing functions on other platforms) to find edges, such as Canny or other operators. Don’t forget that those have a variety of parameters and you may need to experiment with them. BUT: YOU MAY NOT USE ANY HOUGH TOOLS.For example, you need to write your own accumulator array data structures and code for voting and peak finding.

What to submit
--------------

Download and unzip a template for your platform (also under: [https://www.udacity.com/wiki/ud810](https://www.google.com/url?q=https://www.udacity.com/wiki/ud810&sa=D&ust=1520280504008000&usg=AFQjCNGzyPvu56jZadZ0BNpNR6EksXFJpQ)):

*   [ps1\_octave\_template.zip](https://www.google.com/url?q=https://s3.amazonaws.com/content.udacity-data.com/courses/ud810/assignments/ps1_octave_template.zip&sa=D&ust=1520280504009000&usg=AFQjCNEHX8KlTeYb1QcuGeV8SEZvXxDosg)
*   [ps1\_matlab\_template.zip](https://www.google.com/url?q=https://s3.amazonaws.com/content.udacity-data.com/courses/ud810/assignments/ps1_matlab_template.zip&sa=D&ust=1520280504010000&usg=AFQjCNEZEhDYcrX29SIRa_TN6O9lnJFyeg)
*   Note: Python template is being developed, will be announced on Piazza when ready.

Rename it to ps1\_xxxx (i.e. ps1\_matlab, ps1\_octave, or ps1\_python) and add in your solutions:

ps1_xxxx/

*   input/ - input images, videos or other data supplied with the problem set

*   ps1-input0.png
*   ps1-input0-noise.png
*   ps1-input1.png
*   ps1-input2.png
*   ps1-input3.png

*   output/ - directory containing output images and other files your code generates  
    Note: Output images must be stored with following mandatory naming convention:

ps<problem set #>-<question #>-<part>-<counter>.png

Example: ps1-1-a-1.png (first output image for question 1-a)

*   ps1.m or ps1.py - code for completing each part, esp. function calls; all functions themselves must be defined in individual function files with filename same as function name, as indicated
*   *.m or *.py - Matlab/Octave function files (one function per file), Python modules, any utility code
*   ps1_report.pdf - a PDF file that shows all your output for the problem set, including images labeled appropriately (by filename, e.g. ps1-1-a-1.png) so it is clear which section they are for and the small number of written responses necessary to answer some of the questions (as indicated).

Zip it as ps1_xxxx.zip, and submit on T-Square.

Questions
---------

1.  For this question we will use input/ps1-input0.png:

![](https://lh4.googleusercontent.com/8sVZP1IFWKhLAm3EkOMSpFGY4aRGUk8lcNBxPNv-iyhjli0db0JSprMmLsS-axUOJipExEa3MPFNBGxAut4gLQXMhO7S7IlCY1pH4sujfklUmPDZT2f5Umd5m_2Vf3Tu9BuNvP4M-yiDOGwx)

This is a test image for which the answer should be clear, where the “object” boundaries are only lines.

1.  Load the input grayscale image (input/ps1-input0.png) as img and generate an edge image – which is a binary image with white pixels (1) on the edges and black pixels (0) elsewhere.  
      
    For reference, do “doc edge” in Matlab and read about edge operators. Useone operator of your choosing – for this image it probably won’t matter much. If your edge operator uses parameters (like ‘canny’) play with those until you get the edges you would expect to see.  
      
    Output: Store edge image (img_edges) as ps1-1-a-1.png

2.  Implement a Hough Transform method for finding lines. Note that the coordinate system used is as pictured below with the origin placed one pixel above and to the left of the upper-left pixel of the image and with the Y-axis pointing downwards.  
    ![](https://docs.google.com/drawings/d/svoULofii_uzdENi5mXt_eA/image?rev=2&h=281&w=306&ac=1)Thus, the pixel at img(r,c)corresponds to the (x,y)coordinates (r,c), i.e. x=cand y=r. This pixel should vote for line parameters (ρ,θ)where: ρ = x⋅cos(θ) + y⋅sin(θ),and θ = atan2(y,x).  
    ![](https://docs.google.com/drawings/d/snzbgfXkpqOSb1uL3AUDjVw/image?rev=96&h=255&w=297&ac=1)  
    This has the effect of making the positive angular direction clockwise instead of counter-clockwise in the usual convention. Theta (θ) = zero still points in the direction of the positive X-axis.

1.  Write a function hough\_lines\_accthat computes the Hough Transform for lines and produces an accumulator array. Your code should conform to the specifications of the Matlab function hough: [http://www.mathworks.com/help/images/ref/hough.html](https://www.google.com/url?q=http://www.mathworks.com/help/images/ref/hough.html&sa=D&ust=1520280504017000&usg=AFQjCNHQcCs-Rd-TmlEI2aICRg9RE8qO4A)  
    Note that it has two optional parameters RhoResolution and Theta, and returns three values - the hough accumulator array H, theta (θ) values that correspond to columns of H and rho (ρ) values that correspond to rows of H.  
      
    Apply it to the edge image (img_edges) from question 1:  
            \[H, theta, rho\] = hough\_lines\_acc(img_edges);  
    Or, with one optional parameter specified (θ= integers -90to 89, i.e. 180values including 0):  
            \[H, theta, rho\] = hough\_lines\_acc(img_edges, 'Theta', -90:89);  
      
    Function file: hough\_lines\_acc.mcontaining function hough\_lines\_acc (identical name)  
    Output: Store the hough accumulator array (H) as ps1-2-a-1.png (note: write a normalized uint8 version of the array so that the minimum value is mapped to 0 and maximum to 255).  
    
2.  Write a function hough_peaksthat finds indices of the accumulator array (here line parameters) that correspond to local maxima. Your code should conform to the specifications of the Matlab function houghpeaks:  
    [http://www.mathworks.com/help/images/ref/houghpeaks.html](https://www.google.com/url?q=http://www.mathworks.com/help/images/ref/houghpeaks.html&sa=D&ust=1520280504020000&usg=AFQjCNHRfMm2YTy3t1npa_Rg0D4SvP5h5w)Note that you need to return a Qx2 matrix with row indices (here rho) in column 1, and column indices (here theta) in column 2. (This could be used for other peak finding purposes as well.)  
      
    Call your function with the accumulator from the step above to find up to 10 strongest lines:  
            peaks = hough_peaks(H, 10);  
      
    Function file: hough_peaks.m  
    Output: ps1-2-b-1.png \- like above, with peaks highlighted (you can use drawing functions).  
    
3.  Write a function hough\_lines\_drawto draw color lines that correspond to peaks found in the accumulator array. This means you need to look up rho, theta values using the peak indices, and then convert them (back) to line parameters in cartesian coordinates (you can then use regular line-drawing functions).  
      
    Use this to draw lineson the original grayscale(not edge) image. The lines should extend to the edges of the image (aka infinite lines):  
            hough\_lines\_draw(img, 'ps1-2-c-1.png', peaks, rho, theta);  
      
    Function file: hough\_lines\_draw.m  
    Output: ps1-2-c-1.png\- can be saved as a plot directly from hough\_lines\_draw().  
    It should looksomething like this:

![](https://lh5.googleusercontent.com/SZsKhe8dKo4h22bgOogIU9VnjxUlM513n6ahz3u6GHJHcBcbDmorKJGl4n36Nq6qzbsGw7O4rM5bQQmDyXBBjOM7qtC9e9MHQkA4HX8DN54u7Pfb4iqH4owyPArKjrugLqYDKCEaeWDWsTHh)

You might get lines at the boundary of the image too depending upon the edge operator you selected (but those really shouldn’t be there).

4.  What parameters did you use for finding lines in this image?  
    Output: Text response describing your accumulator bin sizes, threshold and neighborhood size parameters for finding peaks, and why/how you picked those.  
    

3.  Now we’re going to add some noise.

1.  Use ps1-input0-noise.png \- same image as before, but with noise. Compute a modestly smoothed version of this image by using a Gaussian filter. Make σ at least a few pixels big.  
    Output: Smoothed image: ps1-3-a-1.png  
    
2.  Using an edge operator of your choosing, create a binary edge image for both the original image (ps1-input0-noise.png) and the smoothed version above.  
    Output: Two edge images: ps1-3-b-1.png(from original), ps1-3-b-2.png (from smoothed)  
    
3.  Now apply your Hough method to the smoothed version of the edge image. Your goal is to adjust the filtering, edge finding, and Hough algorithms to find the lines as best you can in this test case.

Output:\- Hough accumulator array image with peaks highlighted: ps1-3-c-1.png   
\- Intensity image (original one with the noise) with lines drawn on them: ps1-3-c-2.png  
\- Text response:Describe what you had to do to get the best result you could.

4.  For this question use: ps1-input1.png

1.  This image has objects in it whose boundaries are circles (coins) or lines (pens). For this question  you’re still finding lines. Load/createa monochrome version of the image (you can pick a single color channel or use a built-in color to grayscale conversion function), and compute a modestly smoothed version of this image by using a Gaussian filter. Make σ at least a few pixels big.  
    Output: Smoothed monochrome image:ps1-4-a-1.png  
    
2.  Create an edge image for the smoothed version above.  
    Output: Edge image:ps1-4-b-1.png  
    
3.  Apply your Hough algorithm to the edge image to find lines along the pens. Draw the lines in color on the  original monochrome (not edge) image. The lines can extend to the edges of the image.  
    Output:\- Hough accumulator array image withpeaks highlighted: ps1-4-c-1.png  
    \- Original monochromeimage with lines drawn on it: ps1-4-c-2.png  
    \- Text response: Describe what you had to do to get the best result you could.

5.  Now write a circle finding version of the Hough transform. You can implement either the single point method or the point plus gradient method.WARNING: This part may be hard!!! Leave extra time!  
    If you find your arrays getting too big (hint, hint) you might try make the range of radii very small to start with and see if you can find one size circle. Then maybe try the different sizes.

1.  Implement hough\_circles\_accto compute the accumulator array for a given radius.  
    Using the sameoriginal image (monochrome) as above (ps1-input1.png), smooth it, find the edges (or directly use edge image from 4-b above), and trycalling your function with radius= 20:  
            H = hough\_circles\_acc(img_edges, 20);This should return an accumulator H of the same sizeas the supplied image. Each pixel value of the accumulator array should be proportional to the likelihood of a circle of the given radius being present (centered) at that location. Find circle centers by using the same peak finding function:  
            centers = hough_peaks(H, 10);  
      
    Function file: hough\_circles\_acc.m (hough_peaks.m should already be there)  
    Output:  
    \- Smoothed image: ps1-5-a-1.png (this may be identical to  ps1-4-a-1.png)  
    \- Edge image: ps1-5-a-2.png (this may be identical to  ps1-4-b-1.png)\- Original monochrome image with the circles drawn in color:  ps1-5-a-3.png  
    
2.  Implement a function  find_circles that combines the above two steps, searching for circles within a given radius range, and returns circle centers along with their radii:  
            \[centers, radii\] = find\_circles(img\_edges, \[20 50\]);  
      
    Function file: find_circles.m  
    Output:  
    \- Original monochrome image with the circles drawn in color:  ps1-5-b-1.png  
    \- Text response: Describe what you had to do to find circles.

6.  More realistic images. Now that you have Hough methods working, we’re going to try them on images that have clutter in them \- visual elements that are not part of the objects to be detected. The image to use is ps1-input2.png.

1.  Apply your line finder. Use a smoothing filter and edge detector that seems to work best in terms of finding all the pen edges. Don’t worry (until b) about whether you are finding other lines.  
    Output: Smoothed image you used with the Hough lines drawn on them: ps1-6-a-1.png  
    
2.  Likely the last step found lines that are not the boundaries of the pens. What are the problems present?  
    Output: Text response  
    
3.  Attempt to find only the lines that are the \*boundaries\* of the pen.  Three operations you need to try are better thresholding in finding the lines (look for stronger edges), checking the minimum length of the line, looking for nearby parallel lines.  
    Output: Smoothed image with new Hough lines drawn: ps1-6-c-1.png

7.  Finding circles on the same clutter image (ps1-input2.png).  

1.  Apply your circle finder. Use a smoothing filter that seems to work best in terms of finding all the coins.  
    Output:  the smoothed image you used with the circles drawn on them: ps1-7-a-1.png  
    
2.  Are there any false alarms? How would/did you get rid of them?  
    Output: Textresponse (if you did these steps, mention wherethey are in the code by file, line no. and also includebrief snippets)

8.  Sensitivity to distortion. There is a distorted version of the scene at ps1-input3.png.

1.  Apply the line and circle finders to the distorted image. Can you find lines? Circles?    
    Output: Monochrome image with lines and circles (if any) found: ps1-8-a-1.png  
    
2.  What might you do to fix the circle problem?  
    Output: Text response describing what you might try  
    
3.  EXTRA CREDIT:  Try to fix the circle problem (THIS IS HARD).  
    Output:\- Image that is the best shot at fixing the circle problem, with circles found: ps1-8-c-1.png  
    \- Textresponse describing what tried and what worked best (with snippets).