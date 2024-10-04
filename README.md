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
* Physically-Based Materials (BxDFs)
  * Matte (Perfect Diffuse)
  * Mirror (Perfect Specular)
  * Glass (Specular Reflection + Transmission + Fresnel)

## Feature Descriptions

### Scene Description and Loading 

Scenes in this renderer are described with a basic JSON schema that can specify cameras, materials and object primitives, including Spheres, Cubes and Triangle Meshes (using glTF). For more information about the JSON format, check `INSTRUCTIONS.md` and the scene examples in the scenes folder. 

#### glTF meshes

One node / one mesh glTF files can be loaded, so long as they contain the appropriate vertex attributes: positions, normals and texture coordinates (if textures are also included). 

The [glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html) was referenced for this feature and was of great help! 

#### Albedo and Normal Maps

With the glTF format, albedo and normal map textures are renderable and supported. 

Textures are implemented in CUDA using the `CUDA Texture Object API`. You can see how textures can be loaded into the GPU `here`, which can later be accessed by CUDA kernels `here` (Similar to an OpenGL Sampler2D). It's straighforward to use, however the documentation is sparse. I hope this might be another simple example on the internet to help another budding graphics student. 

### Intel Open Image Denoise

Intel Open Image Denoise is a high-performance open-source library for image noise filtering for ray-tracers. This library is integrated into the renderer and filters out the Monte Carlo sample noise that allows us to render smooth images faster. 

An image that would typically take 10,000 samples to render can be done in merely half the time, with potentially cleaner results. 

To denoise rendered images, the API is fairly straightforward. The denoiser accepts 32-bit floating point channels by default and you can see how it's used `here`. 

[Image Here]

For information, go to the [OIDN homepage](https://www.openimagedenoise.org/).

### Materials (BxDF's)

This renderer supports basic Bidirectional Scattering Distribution Function (BSDF) materials. At a high-level, BSDF's describe how light is scattered from certain materials. Using terminology from [Physically Based Rendering: From Theory to Implementation](https://pbr-book.org/3ed-2018/Reflection_Models) (commonly known as PBRT), BSDF's can generally be a combination of BRDF's (Reflectance Models, i.e. Diffuse and Specular) and BTDF's (Transmission Models, how light passes through translucent materials). The "Glass" Material below is an example of a BSDF (Specular Reflection BSDF + Transmission BTDF), and is based off of `FresnelSpecular` which is described by PBRT. 

#### Perfect Diffuse BRDF & Perfect Specular BRDF

#### Glass (Specular Fresnel BSDF)

### Depth of Field

I found [this blog article](https://pathtracing.home.blog/depth-of-field/) which was used for reference and implementation. Producing depth of field (in layman's terms, ['bokeh' in a photograph](https://www.dropicts.com/how-to-achieve-stunning-bokeh-effect-in-your-photo/)) is done by picking a focal point along a generated ray, and jittering it's origin in the xy directions within an aperture. The focal point is the point along the ray where things will be in focus. Increasing the size of your aperture increases the "blur" effect. 

```md
For each extra feature, you must provide the following analysis:
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
