#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <device_launch_parameters.h>

#include <glm/gtc/matrix_transform.hpp>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h" 
#include "meshio.h"

#define ERRORCHECK 1

#define SORT_BY_MATERIAL 0

#define MESH_TEST 1

#define DEBUG_SKY_LIGHT 1
#define DEBUG_ONE_BOUNCE 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
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
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = nullptr;
static GuiDataContainer* guiData = nullptr;
static glm::vec3* dev_image = nullptr;
static Geom* dev_geoms = nullptr;
static Material* dev_materials = nullptr;
static PathSegment* dev_paths = nullptr;
static ShadeableIntersection* dev_intersections = nullptr;

static thrust::device_ptr<PathSegment> thrust_dev_paths = nullptr; 
static thrust::device_ptr<ShadeableIntersection> thrust_dev_intersections = nullptr;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

#if MESH_TEST
    MeshAttributes mesh; 
    if (!meshio::loadMesh("../scenes/geometry/Avocado.gltf", mesh)) {
      std::cerr << "Failed to load test mesh" << std::endl; 
      std::exit(-1); 
    }

    scene->geoms.clear(); 

    Material test_mat{};

    test_mat.color = glm::vec3(197, 227, 234);  // light blue
    test_mat.color /= 255.f; 
    test_mat.specular.color = glm::vec3(0.);
    test_mat.specular.exponent = 0.; 

    scene->materials.push_back(test_mat); 

    glm::mat4 model = glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    
    for (size_t idx = 0; idx < mesh.indices.size(); idx += 3) {
      Geom tri; 
      tri.type = GeomType::TRIANGLE; 
      tri.trianglePos[0] = mesh.positions[mesh.indices[idx + 0]];
      tri.trianglePos[1] = mesh.positions[mesh.indices[idx + 1]];
      tri.trianglePos[2] = mesh.positions[mesh.indices[idx + 2]];

      tri.trianglePos[0] *= 200.f;
      tri.trianglePos[1] *= 200.f;
      tri.trianglePos[2] *= 200.f;

      tri.trianglePos[0].y -= 0.;
      tri.trianglePos[1].y -= 0.;
      tri.trianglePos[2].y -= 0.;

      tri.trianglePos[0] = glm::vec3(model * glm::vec4(tri.trianglePos[0], 1.f));
      tri.trianglePos[1] = glm::vec3(model * glm::vec4(tri.trianglePos[1], 1.f));
      tri.trianglePos[2] = glm::vec3(model * glm::vec4(tri.trianglePos[2], 1.f));

      tri.materialid = scene->materials.size() - 1;    // white diffuse

      scene->geoms.push_back(tri); 
      //break; 
    }

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
#else
    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
#endif

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
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created

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

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);

        // TODO: implement antialiasing by jittering the ray
        float jitterX = u01(rng) - 0.5f;  // Random offset in X ([-0.5, 0.5])
        float jitterY = u01(rng) - 0.5f;  // Random offset in Y ([-0.5, 0.5])

        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitterX)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitterY)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.isFinished = false; 
        segment.isTerminated = false; 
    }
}

//extern MeshAttributes testMesh; 

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == TRIANGLE) {
              glm::vec3 bary; 


              if (glm::intersectRayTriangle(pathSegment.ray.origin, pathSegment.ray.direction, geom.trianglePos[0], geom.trianglePos[1], geom.trianglePos[2], bary)) {

                // We don't use this but it works
                // tmp_intersect = bary.x * geom.trianglePos[0] + bary.y * geom.trianglePos[1] + (1.0f - bary.x - bary.y) * geom.trianglePos[2];

                t = bary.z;

                // calculate tmp normal
                tmp_normal = glm::normalize(glm::cross((geom.trianglePos[0] - geom.trianglePos[2]), (geom.trianglePos[1] - geom.trianglePos[2])));
                if (glm::dot(pathSegment.ray.direction, tmp_normal) > 0) {
                  tmp_normal = -tmp_normal; // Flip the normal if the ray is hitting the backface
                }
              }
              else {
                t = -1;
              }
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
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
        }
    }
}

__device__ glm::vec3 squareToHemisphereCosine(glm::vec2 xi) {
  float x = xi.x;
  float y = xi.y;
  if (x == 0 && y == 0)
    return glm::vec3(0, 0, 0);

  float phi = 0.f;
  float radius = 1.f;
  float a = (2.f * x) - 1.f;
  float b = (2.f * y) - 1.f;

  // Uses squares instead of absolute values
  if ((a * a) > (b * b)) {
    // Top half
    radius *= a;
    phi = (PI / 4) * (b / a);
  }
  else {
    // Bottom half
    radius *= b;
    phi = (PI / 2) - ((PI / 4) * (a / b));
  }

  // Map the distorted Polar coordinates (phi,radius)
  // into the Cartesian (x,y) space
  glm::vec3 disc(0.f, 0.f, 0.f);
  disc.x = glm::cos(phi) * radius;
  disc.y = glm::sin(phi) * radius;

  // I think this ensures this is a hemisphere and not a sphere ? 
  disc.z = glm::sqrt(1.f - (disc.x * disc.x) - (disc.y * disc.y));

  return disc;
}

__device__ glm::vec3 localToWorld(const glm::vec3& normal, const glm::vec3& vec) {
  glm::vec3 tangent, bitangent;

  // create coordinate system from normal 
  if (glm::abs(normal.x) > glm::abs(normal.y)) {
    tangent = glm::vec3(-normal.z, 0, normal.x) / glm::sqrt(normal.x * normal.x + normal.z * normal.z);
  }
  else {
    tangent = glm::vec3(0, normal.z, -normal.y) / glm::sqrt(normal.y * normal.y + normal.z * normal.z);
  }
    
  bitangent = glm::cross(normal, tangent);

  return glm::mat3(glm::normalize(tangent), glm::normalize(bitangent), glm::normalize(normal)) * vec; 
}

__device__ glm::vec3 worldToLocal(const glm::vec3& normal, const glm::vec3& vec) {
  glm::vec3 tangent, bitangent;

  // create coordinate system from normal 
  if (glm::abs(normal.x) > glm::abs(normal.y)) {
    tangent = glm::vec3(-normal.z, 0, normal.x) / glm::sqrt(normal.x * normal.x + normal.z * normal.z);
  }
  else {
    tangent = glm::vec3(0, normal.z, -normal.y) / glm::sqrt(normal.y * normal.y + normal.z * normal.z);
  }

  bitangent = glm::cross(normal, tangent);

  return glm::transpose(glm::mat3(glm::normalize(tangent), glm::normalize(bitangent), glm::normalize(normal))) * vec;
}


__global__ void shadeMaterial(
  int iter,
  int num_paths,
  ShadeableIntersection* shadeableIntersections,
  PathSegment* pathSegments,
  Material* materials,
  int maxBounces)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= num_paths) {
    return; 
  }

  PathSegment pathSegment = pathSegments[idx];

  if (pathSegment.isFinished) {
    return; 
  }

  ShadeableIntersection intersection = shadeableIntersections[idx];

  if (intersection.t <= 0.0f) {
#if DEBUG_SKY_LIGHT
    pathSegment.color *= maxBounces == pathSegment.remainingBounces ? glm::vec3(0.) : glm::vec3(0.7); 
    pathSegment.isFinished = true; 
#else
    pathSegment.color = glm::vec3(0.0f);
    pathSegment.isTerminated = true; 
#endif
  } 
  else if (pathSegment.remainingBounces <= 0) {
    pathSegment.color = glm::vec3(0.0f);
    pathSegment.isTerminated = true; 
  }
  else {
    Material material = materials[intersection.materialId];
    float pdf = 0.f;
    glm::vec2 xi; 

    // compute a random vec2
    thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, pathSegment.remainingBounces);
    thrust::uniform_real_distribution<float> u01(0, 1);
    xi.x = u01(rng); 
    xi.y = u01(rng); 
    if (material.emittance > 0.0f) {                      // LIGHT
      pathSegment.color *= (material.color * material.emittance);
      pathSegment.isFinished = true; 
    }
    else if (material.hasReflective > 0.f) {              // SPECULAR
      glm::vec3 wo = glm::normalize(pathSegment.ray.direction); 
      glm::vec3 wi = glm::normalize(glm::reflect(wo, intersection.surfaceNormal)); 

      pathSegment.color *= material.color; 

      // new ray for the next bounce
      pathSegment.ray.origin = pathSegment.ray.origin + (intersection.t * pathSegment.ray.direction);
      pathSegment.ray.origin += EPSILON * wi;   // slightly offset the ray origin in the direction of the ray direction
      pathSegment.ray.direction = wi;

      --pathSegment.remainingBounces; 
    }
    else {                                                // DIFFUSE

      // generate random direction in hemisphere
      glm::vec3 wi = squareToHemisphereCosine(xi);

      // get the pdf (square to hemisphere cosine)
      pdf = glm::abs(wi.z) * INV_PI;

      if (pdf < EPSILON || isnan(pdf)) {
        pathSegment.isTerminated = true; 
        pathSegments[idx] = pathSegment;
        return; 
      }

#if MESH_TEST
      glm::vec3 bsdfValue = ((0.5f * intersection.surfaceNormal) + 1.f) * INV_PI;
#else
      glm::vec3 bsdfValue = material.color * INV_PI;
#endif

      // convert vec3 into the world coordinate system (using surface normal)
      wi = glm::normalize(localToWorld(intersection.surfaceNormal, wi));

      // update throughput
      pathSegment.color *= bsdfValue * glm::abs(glm::dot(wi, intersection.surfaceNormal)) / pdf;
      pathSegment.color = glm::clamp(pathSegment.color, 0.f, 1.f); 

      // new ray for the next bounce
      pathSegment.ray.origin = pathSegment.ray.origin + (intersection.t * pathSegment.ray.direction); 
      pathSegment.ray.origin += EPSILON * wi;   // slightly offset the ray origin in the direction of the ray direction
      pathSegment.ray.direction = wi; 

      --pathSegment.remainingBounces;
    }
  }

  // read back into global memory
  pathSegments[idx] = pathSegment;
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
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
    const int traceDepth = 1; // DEBUG
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
            traceDepth
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
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
