CUDA Path Tracer
================

> University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 1 - Flocking
> * Michael Mason
>   + [Personal Website](https://www.michaelmason.xyz/)
> * Tested on: Windows 11, Ryzen 9 5900HS @ 3.00GHz 16GB, RTX 3080 (Laptop) 8192MB 

## Overview

This is a Path Tracing Renderer (i.e. Arnold, RenderMan, V-Ray) written in C++ and CUDA which utilizes GPU hardware. This is an interactive renderer! It supports basic materials and scene handling, including glTF support, color & normal textures, and more! See below. 

Supported Features: 

* glTF File Format
* Albedo Maps
* Normal Maps
* Depth of field
* Open Image Denoiser intergration
* Physically-Based Materials (BSDFs)
  * Matte (Perfect Diffuse)
  * Mirror (Perfect Specular)
  * Glass (Specular Reflection + Transmission + Fresnel)

## Feature Descriptions

### Scene Loading 

#### glTF meshes

#### Albedo and Normal Maps

### Open Image Denoiser

### Materials

#### Perfect Diffuse & Perfect Specular

#### Glass (Specular Fresnel)

For each extra feature, you must provide the following analysis:

```md
From the examples I have seen, this isn't followed verbatim...

* Overview write-up of the feature along with before/after images.
* Performance impact of the feature.
* If you did something to accelerate the feature, what did you do and why?
* Compare your GPU version of the feature to a HYPOTHETICAL CPU version (you don't have to implement it!). Does it benefit or suffer from being implemented on the GPU?
* How might this feature be optimized beyond your current implementation?
```

## Performance Analysis

### Stream Compaction
```md
Stream compaction helps most after a few bounces. Print and plot the effects of stream compaction within a single iteration (i.e. the number of unterminated rays after each bounce) and evaluate the benefits you get from stream compaction.
```
### Open & Closed Scenes 

### Effects of stream compaction
```
Compare scenes which are open (like the given cornell box) and closed (i.e. no light can escape the scene). Again, compare the performance effects of stream compaction! Remember, stream compaction only affects rays which terminate, so what might you expect?
```

```md
For optimizations that target specific kernels, we recommend using stacked bar graphs to convey total execution time and improvements in individual kernels. For example:

  ![Clearly the Macchiato is optimal.](img/stacked_bar_graph.png)

Timings from NSight should be very useful for generating these kinds of charts.
```
