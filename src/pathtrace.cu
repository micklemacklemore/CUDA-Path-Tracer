#include "pathtrace.h"
#include "sceneStructs.h"
#include "scene.h"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h" 
#include "meshio.h"

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>
#include <glm/gtc/matrix_transform.hpp>
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include <OpenImageDenoise/oidn.hpp>

#include <cstdio>
#include <cuda.h>
#include <cmath>

#define ERRORCHECK 1

#define SORT_BY_MATERIAL 0
#define STREAM_COMPACTION 0 // TODO

//#define MESH_TEST 0
//#define DEBUG_SKY_LIGHT 0
//#define DEBUG_SKY_LIGHT_BLACK_BG 1
//#define DEBUG_NORMALS 0

#define DEPTH_OF_FIELD 0
#define OIDN 0

#define DEBUG_ONE_BOUNCE 0
#define FORCE_NUM_BOUNCES 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

static Scene* hst_scene = nullptr;
static GuiDataContainer* guiData = nullptr;
static glm::vec3* dev_image = nullptr;
static glm::vec3* dev_finalimage = nullptr; 
static Geom* dev_geoms = nullptr;
static Triangle* dev_tris = nullptr; 
static Material* dev_materials = nullptr;
static PathSegment* dev_paths = nullptr;
static ShadeableIntersection* dev_intersections = nullptr;
// need to keep a versions of cuda tex objects and cuda arrays in host so that we can still free it later
static std::vector<cudaTextureObject_t> host_textures; 
static std::vector<cudaArray_t> host_cuArray;  
static cudaTextureObject_t* dev_textures = nullptr; 
static thrust::device_ptr<PathSegment> thrust_dev_paths = nullptr; 
static thrust::device_ptr<ShadeableIntersection> thrust_dev_intersections = nullptr;

extern std::unique_ptr<oidn::DeviceRef> device; 

void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess == err)
  {
    return;
  }
  fprintf(stderr, "CUDA error");
  if (file)
  {
    fprintf(stderr, " (%s:%d)", file, line);
  }
  fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
  getchar();
#endif // _WIN32
  exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
  int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
  return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y)
  {
    int index = x + (y * resolution.x);
    glm::vec3 pix = image[index];

    glm::ivec3 color;
#if OIDN
    color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
    color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
    color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);
#else
    color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
    color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
    color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);
#endif

    // Each thread writes one pixel location in the texture (textel)
    pbo[index].w = 0;
    pbo[index].x = color.x;
    pbo[index].y = color.y;
    pbo[index].z = color.z;
  }
}

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

// load imageData into cuda device and get back device pointers to the image
void loadTexturesToCUDADevice(cudaArray_t& cuArray, cudaTextureObject_t& texObj, const meshio::ImageData& imageData) {
  // Allocate CUDA array in device memory
  // channel format desc specifies the number of bits 
  // in each texture channel (rgba) and the component
  // format (in this case floats)
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

  // similar to `cudaMalloc` but for 2D/3D arrays afaik
  cudaMallocArray(&cuArray, &channelDesc, imageData.width, imageData.height);
  checkCUDAError("cudaMallocArray");

  // Set pitch of the source (the width in memory in bytes of the 2D array pointed
  // to by src, including padding), we dont have any padding
  const size_t spitch = imageData.width * 4 * sizeof(float);
  cudaMemcpy2DToArray(
    cuArray, 
    0, 
    0, 
    imageData.buffer.data(),
    spitch, 
    imageData.width * 4 * sizeof(float),
    imageData.height, 
    cudaMemcpyHostToDevice
  );
  checkCUDAError("cudaMemcpy2DToArray");

  // Specify texture "resource"
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);
  checkCUDAError("cudaCreateTextureObject");
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_finalimage, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_finalimage, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    for (const auto& tex : scene->textures) {
      cudaArray_t cuArray;
      cudaTextureObject_t texObj = 0;

      loadTexturesToCUDADevice(cuArray, texObj, tex);

      host_cuArray.push_back(cuArray);
      host_textures.push_back(texObj);
    }

    cudaMalloc(&dev_textures, host_textures.size() * sizeof(cudaTextureObject_t)); 
    cudaMemcpy(dev_textures, host_textures.data(), host_textures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_tris, scene->tris.size() * sizeof(Triangle)); 
    cudaMemcpy(dev_tris, scene->tris.data(), scene->tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice); 

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    thrust_dev_paths = thrust::device_pointer_cast<PathSegment>(dev_paths); 
    thrust_dev_intersections = thrust::device_pointer_cast<ShadeableIntersection>(dev_intersections); 

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_finalimage); 
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_textures);
    cudaFree(dev_tris); 
    
    for (auto& cuArray : host_cuArray) {
      cudaFreeArray(cuArray);
    }

    for (auto& texObj : host_textures) {
      cudaDestroyTextureObject(texObj);
    }

    host_textures.clear(); 
    host_cuArray.clear(); 

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // anti-alias jitter
        float jitterX = (u01(rng) - 0.5f) * cam.pixelLength.x * 150.f;  // we multiply by some constant until it looks good?
        float jitterY = (u01(rng) - 0.5f) * cam.pixelLength.y * 150.f;

        glm::vec3 direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitterX) 
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitterY)
        );

        // Depth of field
#if DEPTH_OF_FIELD
        float focalLength = 7.f;
        float blur = 0.4f;
        float apertureX = (u01(rng) - 0.5f) * blur;
        float apertureY = (u01(rng) - 0.5f) * blur;
        glm::vec3 convergence = cam.position + (focalLength * direction);

        segment.ray.origin = cam.position + glm::vec3(apertureX, apertureY, 0.f); 
        segment.ray.direction = glm::normalize(convergence - segment.ray.origin); 
#else
        segment.ray.origin = cam.position; 
        segment.ray.direction = direction; 
#endif

        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.isFinished = false; 
        segment.isTerminated = false; 
    }
}


__device__ float meshIntersectionTest(const Geom& geom, const Triangle* tris, const Ray& ray, glm::vec3& normal, glm::vec2& uvSample) {
  float t = -1.f;
  float t_min = FLT_MAX;
  bool somethingWasHit = false; 

  glm::vec2 tmp_uvSample;
  glm::vec3 tmp_normal;

  glm::vec3 tmp_intersect; // dummy variable
  if (!HitBoundingBox(geom.minBoundingBox, geom.maxBoundingBox, ray.origin, ray.direction, tmp_intersect)) {
    return -1; 
  }

  for (int i = 0; i < geom.triNum; ++i) {
    glm::vec3 bary;
    Triangle tri = tris[geom.triStart + i]; 

    if (glm::intersectRayTriangle(ray.origin, ray.direction, tri.trianglePos[0], tri.trianglePos[1], tri.trianglePos[2], bary)) {
      t = bary.z;

      // calculate normal
      tmp_normal = bary.x * tri.triangleNor[1] + bary.y * tri.triangleNor[2] + (1.0f - bary.x - bary.y) * tri.triangleNor[0];
      tmp_normal = glm::normalize(tmp_normal);

      // calculate texture sample
      tmp_uvSample = bary.x * tri.triangleTex[1] + bary.y * tri.triangleTex[2] + (1.0f - bary.x - bary.y) * tri.triangleTex[0];

      // TODO: this is supposed to ensure double sidedness, but it seems to do nothing?!
      //if (glm::dot(ray.direction, tmp_normal) > 0) {
      //  tmp_normal = -tmp_normal; // Flip the normal if the ray is hitting the backface
      //}
    }
    else {
      t = -1.f; 
    }

    // Compute the minimum t from the intersection tests to determine what
    // scene geometry object was hit first.
    if (t > 0.0f && t_min > t)
    {
      t_min = t;
      normal = tmp_normal;
      uvSample = tmp_uvSample;
      somethingWasHit = true; 
    }
  }

  return somethingWasHit ? t_min : -1.f; 
}

__global__ void computeIntersections(
  int depth,
  int num_paths,
  PathSegment* pathSegments,
  Geom* geoms,
  int geoms_size,
  Triangle* tris,
  ShadeableIntersection* intersections
)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point; // not used?
        bool outside = true;  // not used?
        glm::vec3 normal;
        glm::vec2 uvSample; 
        float t_min = FLT_MAX;
        int hit_geom_index = -1;

        glm::vec3 tmp_intersect;  // not used?
        glm::vec2 tmp_uvSample; 
        glm::vec3 tmp_normal;
        bool tmp_outside = true; 

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside);
            }
            else if (geom.type == MESH) 
            {
                t = meshIntersectionTest(geom, tris, pathSegment.ray, tmp_normal, tmp_uvSample); 
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                normal = tmp_normal;
                uvSample = tmp_uvSample; 
                intersect_point = tmp_intersect;  // not used?
                outside = tmp_outside; 
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].texSample = uvSample; 
            intersections[path_index].outside = outside; 
        }
    }
}

__global__ void shadeMaterial(
  int iter,
  int num_paths,
  ShadeableIntersection* shadeableIntersections,
  PathSegment* pathSegments,
  Material* materials,
  int maxBounces, 
  const cudaTextureObject_t* textures
  )
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_paths) {
    return; 
  }

  PathSegment pathSegment = pathSegments[idx];

  // isFinished means "we hit a light and we're done"
  if (pathSegment.isFinished) {
    return; 
  }

  ShadeableIntersection intersection = shadeableIntersections[idx];

  if (intersection.t <= 0.0f) {
    pathSegment.color = glm::vec3(0.0f);
    pathSegment.isTerminated = true; 
  } 
  else if (pathSegment.remainingBounces <= 0) {
    pathSegment.color = glm::vec3(0.0f);
    pathSegment.isTerminated = true; 
  }
  else {
    Material material = materials[intersection.materialId];
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
    scatterRay(pathSegment, intersection, material, textures, rng); 
  }

  // read back into global memory
  pathSegments[idx] = pathSegment;
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths, int iter)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

// devide the samples by iterations
__global__ void normalizeSamples(glm::ivec2 resolution, glm::vec3* out_image, glm::vec3* in_image, int iter)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y)
  {
    int index = x + (y * resolution.x);
    out_image[index] = in_image[index] / (float)iter; 
  }
}

struct isTerminated
{
  __host__ __device__
    bool operator()(const PathSegment& path) const
  {
    return path.isTerminated; 
  }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
#if DEBUG_ONE_BOUNCE
    const int traceDepth = FORCE_NUM_BOUNCES; // DEBUG
#else
    const int traceDepth = hst_scene->state.traceDepth;
#endif
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    while (depth < traceDepth)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
        checkCUDAError("cuda memset");
        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_tris,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        // sort paths / intersections by material 
#if SORT_BY_MATERIAL
        thrust::sort_by_key(thrust_dev_intersections, thrust_dev_intersections + num_paths, thrust_dev_paths);
#endif
        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            traceDepth, 
            dev_textures
        );

        auto new_dev_path_end = thrust::remove_if(thrust_dev_paths, thrust_dev_paths + num_paths, isTerminated());
        num_paths = new_dev_path_end - thrust_dev_paths; 

        depth++;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }

        if (num_paths == 0) {
          break;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths, iter);

    ///////////////////////////////////////////////////////////////////////////

    /* Denoise */
#if OIDN
    if (iter == 1 || iter % 10 == 0) {
      const int pixelcount = cam.resolution.x * cam.resolution.y;
      static oidn::BufferRef colorBuf = device->newBuffer(pixelcount * sizeof(glm::vec3));  // Buffer for input color

      normalizeSamples<<<blocksPerGrid2d, blockSize2d>>> (cam.resolution, dev_finalimage, dev_image, iter);

      // Copy the image from GPU to OIDN buffer (host)
      cudaMemcpy(colorBuf.getData(), dev_finalimage, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
      checkCUDAError("cudaMemcpy from dev_image to colorBuf");

      // Prepare the denoiser
      static oidn::FilterRef filter = device->newFilter("RT");  // "RT" filter is for ray tracing
      filter.setImage("color", colorBuf.getData(), oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
      filter.setImage("output", colorBuf.getData(), oidn::Format::Float3, cam.resolution.x, cam.resolution.y);

      // If you had auxiliary buffers for albedo or normal, you would set them here, like so:
      // filter.setImage("albedo", albedoBuf.getData(), oidn::Format::Float3, cam.resolution.x, cam.resolution.y);
      // filter.setImage("normal", normalBuf.getData(), oidn::Format::Float3, cam.resolution.x, cam.resolution.y);

      filter.commit();

      // Execute the denoiser
      filter.execute();

      // Check for errors
      const char* errorMessage;
      if (device->getError(errorMessage) != oidn::Error::None) {
        std::cerr << "Error during denoising: " << errorMessage << std::endl;
      }

      // Copy the denoised image back to the GPU
      cudaMemcpy(dev_finalimage, colorBuf.getData(), pixelcount * sizeof(glm::vec3), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy from denoisedBuf to dev_finalimage");
    }

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_finalimage);
    // Retrieve image from GPU (for output)
    cudaMemcpy(hst_scene->state.image.data(), dev_finalimage, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#else
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#endif
    checkCUDAError("pathtrace");
}
