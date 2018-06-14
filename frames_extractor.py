
# coding: utf-8

import moviepy.editor as mp

import ffmpeg
import os

import glob
import spynet


# resize_vid : Resizes a video to the specified width and height
#
# vidFile : The video file that is to be resized
# w       : The new width
# h       : The new height
#
def resize_vid(vidFile, w, h):
    
    filename, ext = os.path.splitext(vidFile)
    reszFile = filename + '-' + str(h) + 'p' + ext
    
    clip = mp.VideoFileClip(vidFile)
       
    reszClip = clip.resize(newsize=(w, h))
    reszClip.write_videofile(reszFile, progress_bar=False)
    
    return reszFile


# vid_to_imgs : Outputs each video frame to an output directory
#
# inFile : The video input file from which to extract the frames
# vFPS   : The desired frames per second
#
def vid_to_imgs(inFile, vFPS, get_flow = False):
    
    outImg = 'frame%04d.png'
    outDir, ext = os.path.splitext(inFile)

    # create destination directory
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    
    # resize to 480p
    newVid = resize_vid(inFile, 640, 480)
           
    stream = ffmpeg.input(newVid)
    stream = ffmpeg.filter_(stream, 'fps', fps=vFPS, round='up')
    stream = ffmpeg.output(stream, os.path.join(outDir, outImg))

    ffmpeg.run(stream)
    
    # remove resized video file
    os.remove(newVid)

    if(get_flow):
        imagenames = sorted(glob.glob(outDir+"/*.png"))
        for i in range(len(imagenames)-1):
            spynet.get_flow(imagenames[i],imagenames[i+1],
                            imagenames[i].split(".")[0] + ".flo",
                            arguments_strModel="3")



# imgs_to_vid : Outputs a mp4 video file that's composed of the sequenced frames
#
# imgsDir : The directory that containes the image frames
# outFile : The desired name of the output video file. This file name MUST be unique!
#
#           E.G. 'myVideo.mp4' 
#
# vFPS    : The desired frames per second
#
def imgs_to_vid(imgsDir, outFile, vFPS,prefix="frame"):
    
    inImg = prefix + '%04d.jpg'
    
    # create output file
    if os.path.isfile(outVid):
        os.remove(outVid)
        
    stream = ffmpeg.input(os.path.join(imgsDir, inImg))
    stream = ffmpeg.filter_(stream, 'fps', fps=vFPS, round='up')
    stream = ffmpeg.output(stream, outFile)

    ffmpeg.run(stream)


