# Color_Unmixing_Based_Soft_Segmentation
Implementation of the article :
- Yağiz Aksoy, Tunç Ozan Aydin, Aljoša Smolić, and Marc Pollefeys. 2017. Unmixing-Based Soft Color Segmentation for Image Manipulation. ACM Trans. Graph. 36, 2, Article 19 (April 2017), 19 pages. DOI:https://doi.org/10.1145/3002176

## Folders
- ``./assets`` : Contains the images used for testing
- ``./Code`` : Contains the source code
- ``./Code/data`` : Contains binary files of the results of each step of the method
- ``./Code/res`` : Contains the resulting images of each step of the method

## Prerequisits
- Python 3.9
- OpenCV-contrib
- Numpy
- scipy
- matplotlib
- pickle

## Files
- ``distributions.py`` : contains the code of <b>color model estimation</b> step
- ``minimization.py`` : contains the code of <b>color unmixing</b> step
- ``mate_regulariaztion.py`` : contains the code of <b>mate regularization</b> step
- ``color_refinement.py`` : contains the code of <b>color re-estimation</b> step
