# Video neural style transfer

## usage

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

To get started, open up artistic_style_transfer.ipynb and edit videoDir to be the name of the .mp4 video you want to edit, with the '.mp4' removed. If you want to test it on our iguana video, leave it unchanged. 

If you wish to use a new video, you should uncomment 'import frames_extractor" and the line below, but note that it was difficult to get spynet (for optical flow) working, and there may be some debugging steps needed. We have included a pre-processed video folder ("iguana_pan") for your convenience. 

From there, rename the "style_img" to the path to the style image you want to use. For convenience we have included a few style images in the style folder. 


This will output all the frames of a video, to stitch them back together you can use FFMPEG
