#include <device_launch_parameters.h>
#include <thrust/swap.h>

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

/*
* Trowbridge-Reitz distribution functions 
* Copyright(c) 1998-2016 Matt Pharr, Greg Humphreys, and Wenzel Jakob.
* 
* - Only sampling visible area.
* - Assuming an isotropic material (alpha.x == alpha.y)
*   + although I tried "sampling the visible area" and I failed
*/

__device__ float SinTheta(const glm::vec3& w) {
  return sqrtf(fmaxf(0.f, 1.f - (w.z * w.z))); 
}

__device__ float Tan2Theta(const glm::vec3& w) {
  // cos2theta w.z * w.z
  return fmaxf(0.f, 1.f - (w.z * w.z)) / (w.z * w.z); 
}

__device__ float CosPhi(const glm::vec3& w) {
  float sinTheta = SinTheta(w);
  return (sinTheta == 0.f) ? 1.f : glm::clamp(w.x / sinTheta, -1.f, 1.f);
}

__device__ float SinPhi(const glm::vec3& w) {
  float sinTheta = SinTheta(w);
  return (sinTheta == 0.f) ? 0.f : glm::clamp(w.y / sinTheta, -1.f, 1.f);
}

__device__ float TrowbridgeReitz_Lambda(const glm::vec3& w, float alpha) {
  float absTanTheta = abs(SinTheta(w) / w.z);
  if (isinf(absTanTheta)) return 0.f;
  // Compute _alpha_ for direction _w_
  float alphaa = sqrtf((CosPhi(w) * CosPhi(w)) * alpha * alpha + (SinPhi(w) * SinPhi(w)) * alpha * alpha);
  float alpha2Tan2Theta = (alphaa * absTanTheta) * (alphaa * absTanTheta);
  return (-1.f + sqrtf(1.f + alpha2Tan2Theta)) / 2.f;
}

__device__ float TrowbridgeReitz_D(glm::vec3 wh, float alpha) {
  float tan2Theta = Tan2Theta(wh);
  if (isinf(tan2Theta)) return 0.f;
  const float cos4Theta = wh.z * wh.z * wh.z * wh.z;
  float e =
    ( ( CosPhi(wh) * CosPhi(wh) ) / (alpha * alpha) + ( SinPhi(wh) * SinPhi(wh) ) / (alpha * alpha)) * tan2Theta;
  return 1.f / (PI * alpha * alpha * cos4Theta * (1 + e) * (1 + e));
}

__device__ float TrowbridgeReitz_G(glm::vec3 wo, glm::vec3 wi, float alpha) {
  return 1.f / (1.f + TrowbridgeReitz_Lambda(wo, alpha) + TrowbridgeReitz_Lambda(wi, alpha));
}

__device__ float TrowbridgeReitz_RoughnessToAlpha(float roughness) {
  roughness = fmaxf(roughness, (float)1e-3);
  float x = log(roughness);
  return 1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x +
    0.000640711f * x * x * x * x;
}

__device__ void TrowbridgeReitzSample11(float cosTheta, float U1, float U2, float* slope_x, float* slope_y) {
  // special case (normal incidence)
  if (cosTheta > .9999) {
    float r = sqrtf(U1 / (1 - U1));
    float phi = 6.28318530718 * U2;
    *slope_x = r * cosf(phi);
    *slope_y = r * sinf(phi);
    return;
  }

  float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
  float tanTheta = sinTheta / cosTheta;
  float a = 1 / tanTheta;
  float G1 = 2 / (1 + sqrtf(1.f + 1.f / (a * a)));

  // sample slope_x
  float A = 2.f * U1 / G1 - 1.f;
  float tmp = 1.f / (A * A - 1.f);
  if (tmp > 1e10) tmp = 1e10;
  float B = tanTheta;
  float D = sqrtf(
    fmaxf(float(B * B * tmp * tmp - (A * A - B * B) * tmp), 0.f));
  float slope_x_1 = B * tmp - D;
  float slope_x_2 = B * tmp + D;
  *slope_x = (A < 0 || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

  // sample slope_y
  float S;
  if (U2 > 0.5f) {
    S = 1.f;
    U2 = 2.f * (U2 - .5f);
  }
  else {
    S = -1.f;
    U2 = 2.f * (.5f - U2);
  }
  float z =
    (U2 * (U2 * (U2 * 0.27385f - 0.73369f) + 0.46341f)) /
    (U2 * (U2 * (U2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
  *slope_y = S * z * sqrtf(1.f + *slope_x * *slope_x);

  assert(!isinf(*slope_y));
  assert(!isnan(*slope_y));
}

__device__ glm::vec3 TrowbridgeReitzSample(const glm::vec3& wi, float alpha, float U1, float U2) {
  // 1. stretch wi
  glm::vec3 wiStretched = glm::normalize(glm::vec3(alpha * wi.x, alpha * wi.y, wi.z));

  // 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
  float slope_x, slope_y;
  TrowbridgeReitzSample11(wiStretched.z, U1, U2, &slope_x, &slope_y);

  // 3. rotate
  float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
  slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
  slope_x = tmp;

  // 4. unstretch
  slope_x = alpha * slope_x;
  slope_y = alpha * slope_y;

  // 5. compute normal
  return glm::normalize(glm::vec3(-slope_x, -slope_y, 1.));
}

__device__ glm::vec3 TrowbridgeReitz_Sample_wh(const glm::vec3& wo, const glm::vec2& xi, float alpha) {
// code for sampling visible area didn't seem to work...
#ifdef SAMPLE_VISIBLE_AREA 
  bool flip = wo.z < 0.f;
  glm::vec3 wh = TrowbridgeReitzSample(flip ? -wo : wo, alpha, xi.x, xi.y);
  if (flip) wh = -wh;

  return wh;
#else
  glm::vec3 wh;

  float cosTheta = 0;
  float phi = TWO_PI * xi[1];
  // We'll only handle isotropic microfacet materials
  float tanTheta2 = alpha * alpha * xi[0] / (1.0f - xi[0]);
  cosTheta = 1 / sqrt(1 + tanTheta2);

  float sinTheta =
    sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));

  wh = glm::vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

  if (!(wo.z * wh.z > 0)) {
    wh = -wh;
  }

  return wh;
#endif
}

__device__ float TrowbridgeReitz_pdf(const glm::vec3& wo, const glm::vec3& wh, float alpha) {
  return TrowbridgeReitz_D(wh, alpha) * abs(wh.z);
}

/*
*****************************************************************
*/

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

// Fresnel Dielectric adapted from PBRTv3 source code
// Copyright(c) 1998-2016 Matt Pharr, Greg Humphreys, and Wenzel Jakob.
__device__ float fresnelDielectric(float cosThetaI, float etaI, float etaT) {
  cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);
  // Potentially swap indices of refraction
  bool entering = cosThetaI > 0.f;
  if (!entering) {
    thrust::swap(etaI, etaT);
    cosThetaI = abs(cosThetaI);
  }

  // Compute _cosThetaT_ using Snell's law
  float sinThetaI = sqrtf(fmaxf(0.f, 1.f - cosThetaI * cosThetaI));
  float sinThetaT = etaI / etaT * sinThetaI;

  // Handle total internal reflection
  if (sinThetaT >= 1) return 1;
  float cosThetaT = sqrtf(fmaxf(0.f, 1.f - sinThetaT * sinThetaT));
  float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) /
    ((etaT * cosThetaI) + (etaI * cosThetaT));
  float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) /
    ((etaI * cosThetaI) + (etaT * cosThetaT));
  return (Rparl * Rparl + Rperp * Rperp) * 0.5f;
}

// Refract adapted from PBRTv3 source code
// Copyright(c) 1998-2016 Matt Pharr, Greg Humphreys, and Wenzel Jakob.
__device__ bool Refract(const glm::vec3& wi, const glm::vec3& n, float eta, glm::vec3& wt) {
  // Compute $\cos \theta_\roman{t}$ using Snell's law
  float cosThetaI = glm::dot(n, wi);
  float sin2ThetaI = fmaxf(0.f, float(1.f - cosThetaI * cosThetaI));
  float sin2ThetaT = eta * eta * sin2ThetaI;

  // Handle total internal reflection for transmission
  if (sin2ThetaT >= 1.f) return false;
  float cosThetaT = sqrtf(1 - sin2ThetaT);
  wt = eta * -wi + (eta * cosThetaI - cosThetaT) * glm::vec3(n);
  return true;
}

__device__ glm::vec3 faceForward(const glm::vec3& v, const glm::vec3& v2) {
  return (glm::dot(v, v2) < 0.f) ? -v : v;
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
  glm::vec3 materialColor = m.color; 
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

  float fresnelTerm; 
  {
    glm::vec3 wo = worldToLocal(normal, -pathSegment.ray.direction);
    fresnelTerm = fresnelDielectric(wo.z, 1.5f, 1.f);
  }

  if (m.emittance > 0.0f) {                             // EMISSION
    pathSegment.color *= (m.color * m.emittance);
    pathSegment.isFinished = true;
  }
  else if (m.hasReflective > 0.f) {                     // PERFECT SPECULAR REFLECTION
    glm::vec3 woW = glm::normalize(pathSegment.ray.direction);
    glm::vec3 wo = worldToLocal(normal, woW);  // there is no need for going to a local coord space, but it's for my own learning benefit

    // perfect specular direction
    glm::vec3 wi = glm::normalize(glm::reflect(wo, glm::vec3(0.f, 0.f, 1.f)));

    pathSegment.color *= materialColor;
    pathSegment.color = glm::clamp(pathSegment.color, 0.f, 1.f);

    wi = glm::normalize(localToWorld(normal, wi)); 

    // new ray for the next bounce
    pathSegment.ray.origin = pathSegment.ray.origin + (intersection.t * pathSegment.ray.direction);
    pathSegment.ray.origin += EPSILON * wi;   // slightly offset the ray origin in the direction of the ray direction
    pathSegment.ray.direction = wi;

    --pathSegment.remainingBounces;
  }
  else if (m.hasTransmissive > 0.f) {                   // FRESNEL SPECULAR (Glass) 
    glm::vec3 wo = worldToLocal(normal, -pathSegment.ray.direction); 
    glm::vec3 wi; 
    glm::vec3 bsdfValue; 

    float etaA = 1.f; 
    float etaB = 1.5f; 

    float fresnelTerm = fresnelDielectric(wo.z, etaA, etaB); 
    if (xi.x < fresnelTerm) {
      // reflection
      wi = glm::vec3(-wo.x, -wo.y, wo.z); 
      pdf = fresnelTerm; 
      bsdfValue = fresnelTerm * materialColor / abs(wi.z); 
    }
    else 
    {
      // transmission
      bool entering = wo.z > 0;
      float etaI = entering ? etaA : etaB;
      float etaT = entering ? etaB : etaA;

      glm::vec3 normalLocal = glm::vec3(0, 0, 1);

      // ensure normal is on the same side of the incident vector
      normalLocal = glm::dot(normalLocal, wo) < 0.f ? -normalLocal : normalLocal;

      if (!Refract(wo, normalLocal, etaI / etaT, wi)) {
        pathSegment.isTerminated = true;
        return;
      }
      glm::vec3 ft = materialColor * (1.f - fresnelTerm); 

      pdf = 1.f - fresnelTerm; 
      bsdfValue = ft / abs(wi.z); 
    }

    wi = glm::normalize(localToWorld(normal, wi));

    pathSegment.color *= bsdfValue * glm::abs(glm::dot(wi, normal)) / pdf;
    pathSegment.color = glm::clamp(pathSegment.color, 0.f, 1.f);

    // new ray for the next bounce
    pathSegment.ray.origin = pathSegment.ray.origin + (intersection.t * pathSegment.ray.direction);
    pathSegment.ray.origin += 0.01f * wi;   // slightly offset the ray origin in the direction of the ray direction
    pathSegment.ray.direction = wi;

    --pathSegment.remainingBounces; 
  }
  else if (m.isMicrofacet > 0.f) {                                  // MICROFACET REFLECTION (PLASTIC)
    glm::vec3 wo = worldToLocal(normal, -pathSegment.ray.direction);
    glm::vec3 wi;
    glm::vec3 bsdfValue;

    //float alpha = TrowbridgeReitz_RoughnessToAlpha(m.roughness);   // why isn't this used? 
    float alpha = glm::clamp(m.roughness, 0.01f, 1.f); 
    alpha *= alpha; 

    // --------- get the wh, wi and pdf - sample_f() -------------

    if (wo.z == 0.f) {
      pathSegment.isTerminated = true; 
      return;
    }

    // sample the microfacet distribution to get the half vector
    glm::vec3 wh = TrowbridgeReitz_Sample_wh(wo, xi, alpha);

    if (glm::dot(wo, wh) < 0.f) {
      pathSegment.isTerminated = true; 
      return; 
    }

    wi = glm::reflect(-wo, wh);

    // if not in same hemisphere
    if (!(wo.z * wi.z > 0)) {
      pathSegment.isTerminated = true; 
      return; 
    }

    // compute PDF of wi for microfacet reflection
    pdf = TrowbridgeReitz_pdf(wo, wh, alpha) / (4.f * glm::dot(wo, wh)); 

    if (pdf < EPSILON || isnan(pdf)) {
      pathSegment.isTerminated = true;
      return;
    }

    // --------- calculate bsdf value - f() --------------

    float cosThetaO = abs(wo.z); 
    float cosThetaI = abs(wi.z);

    // wh = wi + wo;  // TODO do we need to recalculate wh?  

    if (cosThetaI < EPSILON || cosThetaO < EPSILON) {
      pathSegment.isTerminated = true; 
      return; 
    }

    if (wh.x < EPSILON && wh.y < EPSILON && wh.z < EPSILON) {
      pathSegment.isTerminated = true;
      return;
    }

    wh = glm::normalize(wh);

    // glm::vec3 F(fresnelDielectric(glm::dot(wi, faceForward(wh, glm::vec3(0.f, 0.f, 1.f))), 1.5f, 1.0f)); 
    glm::vec3 F(1.f); 

    bsdfValue = (m.color * TrowbridgeReitz_D(wh, alpha) * TrowbridgeReitz_G(wo, wi, alpha) * F) / (4.f * cosThetaI * cosThetaO); 

    // ---------- finally, do monte carlo and create new ray ---------------

    // convert vec3 into the world coordinate system (using surface normal)
    wi = glm::normalize(localToWorld(intersection.surfaceNormal, wi));

    // update throughput
    pathSegment.color *= bsdfValue / pdf;
    pathSegment.color = glm::clamp(pathSegment.color, 0.f, 1.f);

    // new ray for the next bounce
    pathSegment.ray.origin = pathSegment.ray.origin + (intersection.t * pathSegment.ray.direction);
    pathSegment.ray.origin += EPSILON * wi;   // slightly offset the ray origin in the direction of the ray direction
    pathSegment.ray.direction = wi;

    --pathSegment.remainingBounces;
  }
  else {            // PERFECT DIFFUSE REFLECTION
    // generate random direction in hemisphere
    glm::vec3 wi = squareToHemisphereCosine(xi);

    // get the pdf (square to hemisphere cosine)
    pdf = glm::abs(wi.z) * INV_PI;

    if (pdf < EPSILON || isnan(pdf)) {
      pathSegment.isTerminated = true;
      return;
    }

    glm::vec3 bsdfValue = materialColor * INV_PI;

    // convert vec3 into the world coordinate system (using surface normal)
    wi = glm::normalize(localToWorld(intersection.surfaceNormal, wi));

    // update throughput
    pathSegment.color *= bsdfValue * glm::abs(glm::dot(wi, normal)) / pdf;
    pathSegment.color = glm::clamp(pathSegment.color, 0.f, 1.f);

    // new ray for the next bounce
    pathSegment.ray.origin = pathSegment.ray.origin + (intersection.t * pathSegment.ray.direction);
    pathSegment.ray.origin += EPSILON * wi;   // slightly offset the ray origin in the direction of the ray direction
    pathSegment.ray.direction = wi;

    --pathSegment.remainingBounces;
  }
}
