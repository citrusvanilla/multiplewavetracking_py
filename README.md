# Vision-Based Near-Shore Wave Tracking and Recognition for High Elevation and Aerial Video Cameras (Python, OpenCV)

![Frame Grab](http://i.imgur.com/ev8KUtD.jpg)
This repository contains a program for modeling, detecting, tracking, and recognizing near-shore ocean waves, written in Python 3+ with use of OpenCV 3+ library.

This program is demoed on video clips from several Southern California locations in a [Vimeo video here](http://vimeo.com/citrusvanilla/multiplewavetracking).

## Software and Library Requirements

- Python 3.10+
- Numpy 1.22+
- OpenCV 4.5+
- OpenCV contrib 4.5+

```python
pip install --upgrade pip
pip install -r requirements.txt
```

## Basic start

```python
python mwt.py path/to/my/scene.mp4
```

## A High-Level Overview

This program implements a method of near-shore ocean wave recognition through a common Computer Vision "recognition" workflow for video sequences, and is fast enough to run in realtime.

The general object recognition workflow for video sequences proceeds from detection, to tracking, and then recognition<sup>[1](#myfootnote1)</sup>.
The process in this program is thus:

1. **Preprocessing of video**: background modeling of the maritime environment and foreground extraction of near-shore waves.
2. **Detection of objects**: Identification and localization of waves in the scene.
3. **Tracking of objects**: Identity-preserving localization of detected waves through successive video frames for the capture of wave dynamics.
4. **Recognition of objects**: Classification of waves based on their dynamics.

Wave recognition has uses in higher-level objectives such as automatic wave period, frequency, and size determination, as well as region-of-interest definition for human activity recognition.

## Program Architecture

In accordance with the general comupter vision recognition workflow for objects in videos, the program is split into four modules (preprocessing, detection, tracking, and the 'Wave' class), in addition to main().

## Code Organization

| File                 | Purpose                                                                        |
| -------------------- | ------------------------------------------------------------------------------ |
| mwt.py               | Implements Multiple Wave Tracking routine.                                     |
| mwt_objects.py       | Definition of the Wave class and associated data members and member functions. |
| mwt_preprocessing.py | Preproesses input video through background modeling and foreground extraction. |
| mwt_detection.py     | Detection Wave sections in frames and converts to Wave objects.                |
| mwt_tracking.py      | Tracks and manages wave objects.                                               |
| mwt_io.py            | Handles video input and output, drawing, logs, and standard output.            |
| scenes/              | A directory of sample videos for the Multiple Wave Tracking program.           |

## The Multiple Wave Tracking Model, in Short

Main() implements the recognition workflow from above. The following bullets list the modeling operation employed in this program, and a full discussion on model choices can be found in ["Model Details"](##ModelDetails) below.

- **Preprocessing**: Input frames are downsized by a factor of four for analysis. Background modeling is performed using a Mixture-of-Gaussians model with five Gaussians per pixels and a background history of 300 frames, resulting in a binary image in which background is represented by values of 255 and foreground as 0. A square denoising kernel of 5x5 pixels is applied pixel-wise to the binary image to remove foreground features that are too small to be considered objects of interest.
- **Detection**: Contour-finding is applied to the denoised image to identify all forground objects. These contours are filtered for both area and shape using a contour's moments, resulting in the return of large, oblong shapes in the scene. These contours are converted to Wave objects and passed to the tracking routine.
- **Tracking**: A region-of-interest is defined for each potential wave object in which we expect the wave to exist in successive frames. The wave's representation is captured using simple linear search through the ROI and its dynamics are updated according to center-of-mass measurements.
- **Recognition**: We use two dynamics to determine whether or not the tracked object is indeed a positive instance of a wave: mass and displacement. Mass is calculated by weighting pixels equally and performing a simple count. Displacement is measured by calculating the orthogonal displacement of the wave's center-of-mass relative to its original major axis. We accept an object as a true instance of a wave if its mass and orthogonal displacement exceed user-defined thresholds.

## Data and Assumptions

In order to use tracking inference in the classification of waves, we must use static video cameras (e.g. surveillance cameras) as input to the program. Included in the scene directory are three videos from different scenes that can be used to test the Multiple Wave Tracking program. These videos are 1280 x 720 pixels and encoded with the mp4 codec. Please note that if you use your own videos, you may have to re-encode your videos to play nice with the OpenCV library. A common tool for handling video codecs is the [FFMEG library](https://www.ffmpeg.org/).

As a vision-based project, this program performs best on scenes in which the object of interest (the wave) is sufficiently separated from other objects (i.e. there is no occlusion or superimposition). This assumption is fair for the wave object as ocean physics dictate that near-shore waves generally have consistently delineated periods of inter-arrival times, due to the physical processes of assimilation, imposition, and interference that take place a great distance from shore.

To this end, a camera is able to pronouce this periodicity on the focal plane simply by increasing its own elevation from the ocean surface plane. That is- the higher the elevation of the camera, the better the separation in the frame.

## Launching the Model

This project uses OpenCV version 3.2.0 and Numpy 1.11.2.

You can launch the program from the command line after navigating to the parent directory and passing the path of an input video as the first argument. For example:

> joe_bloggs multiplewavetracking_py $ python mwt.py ./scenes/some_video_with_waves.mp4

You should see output like this:

    Starting analysis of 840 frames...
    .........100.........200........

The program report the current frame in the analysis every 100 frames, as well as the performance of the program.

The program will report simple statistics at the conclusion of analysis, like the following:

    End of video reached successfully.
    4 wave(s) recognized.
    Program performance: 32.4 frames per second.
    Wave #1: ID: 3, Birth: 78, Death: 737, Max Displacement: 85, Max Mass: 3309
    Wave #2: ID: 906, Birth: 513, Death: 1079, Max Displacement: 81, Max Mass: 2953
    ...

## Output Files

A log of the tracking routine is written to "wave_log.json" for a frame-by-frame breakdown of the program.

The output video will contain a visualization of the tracking. This obviously slows down the performance of the program, but is useful for determining user-defined thresholds for the optimal performance of the program. You can find the drawing operation in the "draw()" function in the mwt_io.py source file.

## Model Discussion

### What exactly is a "wave"?

![Examples of Near-Shore Waves](http://i.imgur.com/4ASgzrn.jpg)

When considering the **static** representation of a wave in color space, we make use of the high contrast between a wave that is broken and the surrounding water environment. For our program, a wave is denoted by the presence of sea foam when it has "broken". Foam as a physical object is the trapping of air inside liquid bubbles whose refractive properties give the foam a holistic color approaching white. This is contrasted with the ocean surface that does not have such refractive properties and rather traps light such that its intensity is much lower than that of foam. Therefore, we when use computer vision to search a maritime image for a wave, we are really looking for the signature of a wave in the form of sea foam. It is important to note that one wave object can be represented by many disparate "sections" along the length of the wave.

Further, a "wave" in our case has an assumed behaviour of **dynamic** movement through time. We can take advantage of the fact that ocean waves propogating from a source travel in a direction that is orthogonal (perpendicular) to the plane tangent to its wavefront to simply describe a wave's travel with a 1-demensional value (e.g. "the wave has traveled 50 feet").

This representation of a wave in the time domain allows us to abstract the near-shore wave identification problem into a recognition-through-tracking formulation.

### Preprocessing

![Preprocessed Frame](http://i.imgur.com/NDMIwOF.jpg)

In our videos of maritime environments, waves are surely the largest objects present and thus our downsizing of input videos by a factor of four (or greater) is acceptable. Background modeling, however, for these environments is very difficult endeavor even with static cameras due to the dynamic nature of liquids.

We employ an adaptive Gaussian Mixture Model (MOG)<sup>[2](#myfootnote2)</sup> with five Gaussians per pixel in order to give us the greatest flexibility in accounting for a variety of light and surface refractions. The MOG model will classify as foreground any pixel whose value is calculated to exceed 2.5x the standard deviation of any one of these Gaussians. The Gaussian distributions that constitute a pixel's representation are determined using Maximum Likelihood Estimation (MLE), solved in our case with an Expectation Maximization (EM) algorithm. Although Background Subtraction methods have been developed specifically for maritime environments, a quick review suggests that MOG performs faster than such methods, while having having slightly worse accuracy on tested datasets<sup>[3](#myfootnote3)</sup> (none of which consider a breaking wave to be foreground, however).

A background history of 300 frames (equivalent to 10 seconds in our sample videos) ensures that the background is adaptive to changes in ambient conditions that are gradual, while ensuring that a wave passing through is an infrequent-enough event to be classified as foreground.

Despite the flexibility of the MOG model, there will still be residual errors in background modeling. To account for this, we apply a denoising operation on the frame that employs a mathematical morphology operation known as 'opening', which is a sequence of erosion followed by dilation. This has the effect of sending the residual foreground pixels from the GMM model to the background while retaining the general shape of salient foreground features.

We note in passing that the EM algorithm that is performed on a per-pixel basis contributes significantly to the overall expense of background modeling in the program, estimated to be about 50% of the total CPU resources.

### Detection

![Detection and Filtering](http://i.imgur.com/nNhNtzW.jpg)

The resultant frame from the preprocessing module is a binary image in which the background is represented by one value while the foreground is represented by another. In our case, we are left with an image in which waves are represented as sea foam in the foreground. Detection is intended to localize these shapes and to subject them to thresholding that further eliminates false instances.

We use the contour finding method from Suzuki and Abe<sup>[4](#myfootnote4)</sup> that employs a traditional border tracing algorithm to return shapes and locations. These contours are filtered for area (to eliminate foreground objects that are too small to be considered) and inertia (to eliminate foreground objects whose shape does not match that of a breaking wave). We are left with large, oblong object contours which we convert to Wave objects and pass to the tracking routine.

### Tracking

![Tracking Waves](http://i.imgur.com/OT51ZTr.jpg)

We can take advantage of two assumptions about waves that eliminate our reliance on traditional sample-based tracking methologies and their associated probabilistic components. The first is that waves are highly periodic in arrival and therefore will not exhibit occlusion or superimposition. The second assumption is about the wave's dynamics; specifically, that a wave's movement can be desribed by its displacement orthogonal to the axis along which the wave was first identified in the video sequence. These two assumptions allow us to confidently define a search region in the next frame using just a center-of-mass estimate in the current frame, and reduces our search space for the wave's position in successive frames to a search along one dimension. The reduction in dimensionality of the search space allows us to cheaply and exhaustively search for a global position that describes our tracked wave in successive frames. We do not need to rely on sample-based tracking methods that are susceptible to drift and/or suboptimal identifications.

The tracking routine also manages multiple representations of the same wave through the merging of these multiple sections into one object, as a wave can be constructed of many disparate contours.

### Recognition

![Recognition](http://i.imgur.com/uUTDLj3.jpg)

Tracking allows us to incorporate dynamics into classification of waves. We use two dynamics to determine whether or not the tracked object is indeed a positive instance of a wave: mass and displacement. Mass is calculated by weighting pixels equally and performing a simple count. Displacement is measured by calculating the orthogonal displacement of the wave's center-of-mass relative to its original major axis. We accept an object as a true instance of a wave if its mass and orthogonal displacement exceed user-defined thresholds.

By introducing tracking, we are able to confidently classify waves in videos by combining simple bilevel representations of the waves with cheaply-calculated dynamics. If we were to resort to bilevel detection methods for waves without employing dynamics, our methods would be susceptible to false positives from many sources. Certainly a large boat might be an example of a false positive, but harder examples that should be negatively classified include wave-types that have similar contour representations to ocean waves, including "shorebreak"-type waves that break on the shore, and "whitecapping"-type waves that have the appearance of breaking due to near-shore winds. Neither of these meet our definition of an ocean wave.

#### Footnotes

<a name="myfootnote1">1</a>: [Bodor, Robert, Bennett Jackson, and Nikolaos Papanikolopoulos. "Vision-based human tracking and activity recognition." Proc. of the 11th Mediterranean Conf. on Control and Automation. Vol. 1. 2003.](http://mha.cs.umn.edu/Papers/Vision_Tracking_Recognition.pdf)

<a name="myfootnote2">2</a>: [KaewTraKulPong, Pakorn, and Richard Bowden. "An improved adaptive background mixture model for real-time tracking with shadow detection." Video-based surveillance systems 1 (2002): 135-144.](https://link.springer.com/chapter/10.1007/978-1-4615-0913-4_11)

<a name="myfootnote3">3</a>: [Bloisi, Domenico D., Andrea Pennisi, and Luca Iocchi. "Background modeling in the maritime domain." Machine vision and applications 25.5 (2014): 1257-1269.](https://link.springer.com/article/10.1007/s00138-013-0554-5)

<a name="myfootnote4">4</a>: [Suzuki, Satoshi. "Topological structural analysis of digitized binary images by border following." Computer vision, graphics, and image processing 30.1 (1985): 32-46.](http://www.sciencedirect.com/science/article/pii/0734189X85900167)
