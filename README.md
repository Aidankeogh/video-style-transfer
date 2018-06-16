# Video neural style transfer

## usage

Video: https://drive.google.com/file/d/1v_XHuqZYGCtufy8bKvWbnX-U98ufL_xy/view


This tool requires

(For setup - not recommended)
ffmpeg python - https://github.com/kkroening/ffmpeg-python
Moviepy - https://zulko.github.io/moviepy/

(For style transfer)
Pytorch 0.4 - https://pytorch.org/
Opencv python - https://pypi.org/project/opencv-python/
Numpy - https://pypi.org/project/numpy/
PIL - https://pypi.org/project/PIL/
Scipy - https://pypi.org/project/scipy/

To use a new video, open setup.ipynb, and follow the instructions editing the video directory to the video you want to use. This requires SPyNet to work to create optical flow files between images, which can be difficult to get working, so I have included a pre-processed video in test-videos/iguana_pan for conveience.

To get started on style transfer, open up artistic_style_transfer.ipynb and edit videoDir to be the name of the directory created in setup.ipynb. If you want to test it on our iguana video, leave it unchanged. 


From there, rename the "style_img" to the path to the style image you want to use. For convenience we have included a few style images in the style folder. 


This will output all the frames of a video, to stitch them back together you can use FFMPEG
