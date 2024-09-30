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

#define MESH_TEST 0

#define DEBUG_SKY_LIGHT 0
#define DEBUG_SKY_LIGHT_BLACK_BG 1

#define DEBUG_ONE_BOUNCE 0
#define FORCE_NUM_BOUNCES 1

#define DEBUG_NORMALS 0


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
// need to keep a versions of cuda tex objects and cuda arrays in host so that we can still free it later
static std::vector<cudaTextureObject_t> host_textures;  // TODO: this should really be part of Scene!
static std::vector<cudaArray_t> host_cuArray; // TODO: do i need to free this? 
static cudaTextureObject_t* dev_textures = nullptr; 
static thrust::device_ptr<PathSegment> thrust_dev_paths = nullptr; 
static thrust::device_ptr<ShadeableIntersection> thrust_dev_intersections = nullptr;

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
  cudaMemcpy2DToArray(cuArray, 0, 0, imageData.buffer.data(),
    spitch, imageData.width * 4 * sizeof(float),
    imageData.height, cudaMemcpyHostToDevice);
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
    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

#if MESH_TEST
      meshio::MeshAttributes mesh;
      // if (!meshio::loadMesh("../scenes/geometry/Avocado.gltf", mesh)) {
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
      test_mat.textureIdx.albedo = -1;
      test_mat.textureIdx.normal = -1;

      glm::mat4 model = glm::rotate(glm::mat4(1.0f), glm::radians(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

      if (mesh.textureAlbedo.exists()) {
        cudaArray_t cuArrayAlbedo;
        cudaTextureObject_t texObjAlbedo = 0;

        loadTexturesToCUDADevice(cuArrayAlbedo, texObjAlbedo, mesh.textureAlbedo);

        host_cuArray.push_back(cuArrayAlbedo);
        host_textures.push_back(texObjAlbedo);

        test_mat.textureIdx.albedo = host_textures.size() - 1;
      }

      if (mesh.textureNormal.exists()) {
        cudaArray_t cuArrayNormal;
        cudaTextureObject_t texObjNormal = 0;

        loadTexturesToCUDADevice(cuArrayNormal, texObjNormal, mesh.textureNormal);

        host_cuArray.push_back(cuArrayNormal);
        host_textures.push_back(texObjNormal);

        test_mat.textureIdx.normal = host_textures.size() - 1;
      }

      scene->materials.push_back(test_mat);

      for (size_t idx = 0; idx < mesh.indices.size(); idx += 3) {
        Geom tri;
        tri.type = GeomType::TRIANGLE;
        tri.trianglePos[0] = mesh.positions[mesh.indices[idx]];
        tri.trianglePos[1] = mesh.positions[mesh.indices[idx + 1]];
        tri.trianglePos[2] = mesh.positions[mesh.indices[idx + 2]];

        tri.triangleNor[0] = mesh.normals[mesh.indices[idx]];
        tri.triangleNor[1] = mesh.normals[mesh.indices[idx + 1]];
        tri.triangleNor[2] = mesh.normals[mesh.indices[idx + 2]];

        tri.triangleTex[0] = mesh.texcoords[mesh.indices[idx]];
        tri.triangleTex[1] = mesh.texcoords[mesh.indices[idx + 1]];
        tri.triangleTex[2] = mesh.texcoords[mesh.indices[idx + 2]];

        // for material.gltf

        /*tri.trianglePos[0].y += 5.;
        tri.trianglePos[1].y += 5.;
        tri.trianglePos[2].y += 5.;*/

        // for the avocado

        tri.trianglePos[0] *= 100.f;
        tri.trianglePos[1] *= 100.f;
        tri.trianglePos[2] *= 100.f;

        tri.trianglePos[0] = glm::vec3(model * glm::vec4(tri.trianglePos[0], 1.f));
        tri.trianglePos[1] = glm::vec3(model * glm::vec4(tri.trianglePos[1], 1.f));
        tri.trianglePos[2] = glm::vec3(model * glm::vec4(tri.trianglePos[2], 1.f));

        tri.materialid = scene->materials.size() - 1;

        scene->geoms.push_back(tri);
      }
#endif
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
    cudaFree(dev_textures);
    
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

__global__ void computeIntersections(
  int depth,
  int num_paths,
  PathSegment* pathSegments,
  Geom* geoms,
  int geoms_size,
  ShadeableIntersection* intersections
)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        glm::vec2 uvSample; 
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;
        glm::vec3 bitangent;
        glm::vec3 tangent;

        glm::vec2 tmp_uvSample; 
        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;
        glm::vec3 tmp_bitangent; 
        glm::vec3 tmp_tangent; 

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
                t = bary.z;

                // calculate normal
                tmp_normal = bary.x * geom.triangleNor[1] + bary.y * geom.triangleNor[2] + (1.0f - bary.x - bary.y) * geom.triangleNor[0];
                tmp_normal = glm::normalize(tmp_normal); 
                tmp_uvSample = bary.x * geom.triangleTex[1] + bary.y * geom.triangleTex[2] + (1.0f - bary.x - bary.y) * geom.triangleTex[0];

                // calculate tangents and bitangents (TODO: this should be precalculated)
                glm::vec3 edge1 = geom.trianglePos[1] - geom.trianglePos[0];
                glm::vec3 edge2 = geom.trianglePos[2] - geom.trianglePos[0];
                glm::vec2 deltaUV1 = geom.triangleTex[1] - geom.triangleTex[0]; 
                glm::vec2 deltaUV2 = geom.triangleTex[2] - geom.triangleTex[0];

                //float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
                // Avoid division by zero or near-zero values
                float f = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
                if (fabs(f) < 1e-6) {
                  f = 1.0f; // Prevent degenerate UV triangles
                }
                else {
                  f = 1.0f / f;
                }

                tmp_tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
                tmp_tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
                tmp_tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);

                tmp_bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
                tmp_bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
                tmp_bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);

                // Normalize tangent and bitangent
                tmp_tangent = glm::normalize(tmp_tangent);
                tmp_bitangent = glm::normalize(tmp_bitangent);

                // Orthogonalize tangent to the normal
                tmp_tangent = glm::normalize(tmp_tangent - glm::dot(tmp_tangent, tmp_normal) * tmp_normal);

                // Ensure correct handedness of tangent space
                if (glm::dot(glm::cross(tmp_tangent, tmp_bitangent), tmp_normal) < 0.0f) {
                  tmp_bitangent = -tmp_bitangent;
                }

                // This produces weird results, i'm not sure why
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
                uvSample = tmp_uvSample; 
                tangent = tmp_tangent; 
                bitangent = tmp_bitangent; 
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
            intersections[path_index].tangent = tangent; 
            intersections[path_index].bitangent = bitangent; 
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
  int maxBounces, 
  cudaTextureObject_t* textures
  )
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
#if DEBUG_SKY_LIGHT && DEBUG_SKY_LIGHT_BLACK_BG
    pathSegment.color *= pathSegment.remainingBounces == maxBounces ? glm::vec3(0.) : glm::vec3(1.);
    pathSegment.isFinished = true;
#elif DEBUG_SKY_LIGHT
    pathSegment.color *= glm::vec3(1.); 
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
    float isSpec = u01(rng); 

    if (material.emittance > 0.0f) {                      // LIGHT
      pathSegment.color *= (material.color * material.emittance);
      pathSegment.isFinished = true; 
    }
    else if (material.hasReflective > 0.f && isSpec > 0.5) {              // SPECULAR
      glm::vec3 wo = glm::normalize(pathSegment.ray.direction); 

      // perfect specular direction
      glm::vec3 R = glm::normalize(glm::reflect(wo, intersection.surfaceNormal)); 

      float exponent = 300.f;  // TODO: this is the shininess, should come from the material

      // sample random polar coordinates
      float pol = acosf(powf(xi.x, 1.f / (exponent + 1.f)));
      float azi = 2.f * PI * xi.y; 

      // using a PDF here just did not work for me so I didn't use one

      // convert to cartesian coords
      glm::vec3 wi(cosf(azi) * sinf(pol), sinf(azi) * sinf(pol), cosf(pol));
      wi = glm::normalize(wi); 
      wi = localToWorld(intersection.surfaceNormal, wi); 

      pathSegment.color *= material.color / 0.5f;
      pathSegment.color = glm::clamp(pathSegment.color, 0.f, 1.f);

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

      glm::vec3 bsdfValue = material.color * INV_PI; 
      if (material.textureIdx.albedo != -1) {
        float4 texCol = tex2D<float4>(textures[material.textureIdx.albedo], intersection.texSample.s, intersection.texSample.t); 
        bsdfValue = glm::vec3(texCol.x, texCol.y, texCol.z) * material.color * INV_PI;
      }

      if (material.textureIdx.normal != -1) {
        float4 texNorCol = tex2D<float4>(textures[material.textureIdx.normal], intersection.texSample.s, intersection.texSample.t);
        glm::vec3 normal = glm::vec3(texNorCol.x, texNorCol.y, texNorCol.z);
        normal = (normal * 2.f) - 1.f;
        normal = glm::normalize(localToWorld(intersection.surfaceNormal, normal));
        intersection.surfaceNormal = normal; 
      }

#if DEBUG_NORMALS
      bsdfValue = intersection.surfaceNormal * INV_PI;
#endif
      // convert vec3 into the world coordinate system (using surface normal)
      wi = glm::normalize(localToWorld(intersection.surfaceNormal, wi));

      // update throughput
      pathSegment.color *= bsdfValue * glm::abs(glm::dot(wi, intersection.surfaceNormal)) / pdf; 
      if (material.hasReflective > 0.f) {
        pathSegment.color /= 0.5f; 
      }
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
    finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
