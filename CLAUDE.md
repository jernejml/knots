#TASK

Intro
The goal of this task is to see how well you perform in a practical problem-solving situation. The
problem is relatively straightforward:
You are given many sequential frames of wood boards, captured by a color camera. These frames
are taken in a coordinate space where X is the longitudinal axis of the board, while Y is the width of
the board. A board may start and end within a frame, or the frame is a continuation of the board
already present in the previous frame. You can safely assume that there is no overlap between the
frames and there is no gap either (frames stitched end-to-end are valid representations of entire
boards). Frames are labeled as {boardIndex}_{frameNumber}.png, and annotations follow the same
rule. The task is to develop a program which takes these frames as input, and outputs the polygon
bounds of all knots on each individual board (https://en.wikipedia.org/wiki/Wood#Knots).
We expect this task should not take longer than 14 days.

Requirements
•Provide a docker image within which your program can be built and executed.
•Include a test mode which compares the outputs of your program to already annotated scans
stored in some configurable test directory.
•If using CUDA, use at least version 12.8
•Create a git repository for your program. Note that basic knowledge of Git and some level of
commit organization is expected (eg. having a single commit containing all changes is bad)

Guidelines
•You may use any programming language, though C++ (using CMake as the build system) is
preferred for the final program.
•Use a code formatter to keep style consistent
•Usage of LLMs like ChatGPT is allowed, but in moderation. The point is to get insights into
your own thought process and not the thought process of the LLMs.

Closing remarks
After you are done with the task, let us know and we will schedule a call to go over what you did,
how it works, where you had issues and similar things. Try to have fun doing this challenge and talk
to you soon! :)


Clarification on "overlap":

"no overlap" means "no gap" — frames tile the board with no
missing pixels — not "no redundancy." A pixel-level check shows
consecutive frames overlap by 50%: frame width = 640 px, stride = 320 px.
Each board pixel is captured in two adjacent frames, and each physical
knot is typically annotated once per frame that sees it.

Current approach/plan:

Start by analyzing the dataset to understand frame dimensions and label
format. Pick a stratified train/val/test split by board so small failure
modes (TBD) don't end up only in train.
Convert the rectangle labels to per-frame polygons offline with SAM2
(bbox + center-point prompt). Train a YOLOv11-seg student on those                                                                                                                                                                                                                                                                                                                                                   
polygons. Export the trained model to ONNX as the handoff between Python                                                                                                                                                                                                                                                                                                                                                
training and C++ inference. At runtime, the C++ binary runs per-frame                                                                                                                                                                                                                                                                                                                                                   
inference and stitches the results into per-board polygons: translate                                                                                                                                                                                                                                                                                                                                                   
each polygon by `frame_idx * STRIDE_PX` (frames are 640 px wide but                                                                                                                                                                                                                                                                                                                                                     
advance by only 320 px — see Clarification on "overlap" above) and take                                                                                                                                                                                                                                                                                                                                                 
the raster union via `cv::fillPoly` + `cv::findContours`.

Provided data (images+labels) reside in 'data' directory
