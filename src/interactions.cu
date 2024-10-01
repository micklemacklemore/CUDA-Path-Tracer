#include <device_launch_parameters.h>

#include "interactions.h"

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

__device__ void scatterRay(
    PathSegment & pathSegment,
    ShadeableIntersection& intersection,
    const Material &m,
    const cudaTextureObject_t* textures,
    thrust::default_random_engine &rng)
{
  thrust::uniform_real_distribution<float> u01(0, 1);
  glm::vec2 xi(u01(rng), u01(rng));
  float pdf = 0.0f; 
  glm::vec3 materialColor; 
  glm::vec3 normal = intersection.surfaceNormal; 

  if (m.textureIdx.albedo != -1) {
    float4 texCol = tex2D<float4>(textures[m.textureIdx.albedo], intersection.texSample.s, intersection.texSample.t);
    materialColor = glm::vec3(texCol.x, texCol.y, texCol.z) * m.color;
  }

  if (m.textureIdx.normal != -1) {
    float4 texNorCol = tex2D<float4>(textures[m.textureIdx.normal], intersection.texSample.s, intersection.texSample.t);
    normal = glm::vec3(texNorCol.x, texNorCol.y, texNorCol.z);
    normal = (normal * 2.f) - 1.f;
    normal = glm::normalize(localToWorld(intersection.surfaceNormal, normal));
  }

  if (m.emittance > 0.0f) {                      // LIGHT
    pathSegment.color *= (m.color * m.emittance);
    pathSegment.isFinished = true;
  }
  else if (m.hasReflective > 0.f) {              // PERFECT SPECULAR
    glm::vec3 wo = glm::normalize(pathSegment.ray.direction);

    // perfect specular direction
    glm::vec3 wi = glm::normalize(glm::reflect(wo, normal));

    pathSegment.color *= m.color;
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
      return;
    }

    glm::vec3 bsdfValue = m.color * INV_PI;

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
