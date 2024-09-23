// Define these only in *one* .cc file.
#include "meshio.h"

#include <tiny_gltf.h>
#include <glm/gtc/type_ptr.hpp>

#include <cstdio>

// TODO could use things like GetComponentSizeInBytes to double check raw array

void printVector(const std::string& name, const std::vector<double>& v) {
  if (!v.empty()) {
    std::printf("%s: %f %f %f\n", name.c_str(), v[0], v[1], v[2]);
  }
  else {
    std::printf("%s: empty\n", name.c_str()); 
  }
}

void printMatrix(const std::string& name, const std::vector<double>& m) {
  std::printf("%s: \n", name.c_str());
  if (!m.empty()) {
    // column matrix
    std::printf("%f %f %f %f\n", m[4 * 0], m[4 * 1], m[4 * 2], m[4 * 3]);
    std::printf("%f %f %f %f\n", m[4 * 0 + 1], m[4 * 1 + 1], m[4 * 2 + 1], m[4 * 3 + 1]);
    std::printf("%f %f %f %f\n", m[4 * 0 + 2], m[4 * 1 + 2], m[4 * 2 + 2], m[4 * 3 + 2]);
    std::printf("%f %f %f %f\n", m[4 * 0 + 3], m[4 * 1 + 3], m[4 * 2 + 3], m[4 * 3 + 3]);
  }
}

// assume glTF is 1 node, 1 mesh, 1 primitive... (TODO: make this a bit more robust)
bool meshio::loadMesh(std::string filename, MeshAttributes& out_attr) {
	tinygltf::Model model; 
	tinygltf::TinyGLTF loader;

  std::string err;
  std::string warn;

  bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);

  if (!warn.empty()) {
    std::fprintf(stderr, "Warn: %s\n", warn.c_str());
  }

  if (!err.empty()) {
    std::fprintf(stderr, "Err: %s\n", err.c_str());
  }

  if (!ret) {
    std::fprintf(stderr, "Failed to parse glTF\n");
    return false;
  }

  if (model.meshes.empty()) {
    return false; 
  }

  // primitives contain index for attribute buffers, materials, draw mode
  tinygltf::Primitive& prim = model.meshes[0].primitives[0];

  // must be triangles
  if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
    return false; 
  }

  int posIdx, norIdx; 
  {
    auto it = prim.attributes.find("POSITION");
    posIdx = it == prim.attributes.end() ? -1 : it->second;

    it = prim.attributes.find("NORMAL");
    norIdx = it == prim.attributes.end() ? -1 : it->second;
  }

  // must at least have positions array!
  if (posIdx == -1) {
    return false; 
  } 

  // fill positions array
  {
    tinygltf::Accessor& accessor = model.accessors[posIdx];
    tinygltf::BufferView& bufferview = model.bufferViews.at(accessor.bufferView);   // buffer views index is optional
    
    // We must have vec3's of floats
    if (accessor.type != TINYGLTF_TYPE_VEC3 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
      return false; 
    }

    tinygltf::Buffer& buffer = model.buffers[bufferview.buffer]; 

    size_t offset = bufferview.byteOffset + accessor.byteOffset;
    size_t count = accessor.count; 

    std::vector<glm::float32> pos;
    pos.resize(count * 3); 

    std::memcpy(pos.data(), &buffer.data[offset], count * sizeof(glm::float32) * 3); 

    out_attr.positions.clear();

    for (size_t i = 0; i < count; ++i) {
      out_attr.positions.emplace_back(glm::make_vec3(&pos[i * 3])); 
    }
  }

  // fill normals array
  out_attr.normals.clear(); 
  if (norIdx != -1) {
    tinygltf::Accessor& accessor = model.accessors[norIdx];
    tinygltf::BufferView& bufferview = model.bufferViews.at(accessor.bufferView);   // buffer views index is optional

    // We must have vec3's of floats
    if (accessor.type != TINYGLTF_TYPE_VEC3 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
      return false;
    }

    tinygltf::Buffer& buffer = model.buffers[bufferview.buffer];

    size_t offset = bufferview.byteOffset + accessor.byteOffset;
    size_t count = accessor.count;

    std::vector<glm::float32> nor;
    nor.resize(count * 3);

    std::memcpy(nor.data(), &buffer.data[offset], count * sizeof(glm::float32) * 3);

    for (size_t i = 0; i < count; ++i) {
      out_attr.normals.emplace_back(glm::make_vec3(&nor[i * 3]));
    }
  }

  // fill index array (if it exists)
  out_attr.indices.clear(); 
  if (prim.indices != -1) {
    tinygltf::Accessor& accessor = model.accessors[prim.indices];
    tinygltf::BufferView& bufferview = model.bufferViews.at(accessor.bufferView);   // buffer views index is optional

    // We must have vec3's of floats
    if (accessor.type != TINYGLTF_TYPE_SCALAR || accessor.componentType != TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
      return false;
    }

    tinygltf::Buffer& buffer = model.buffers[bufferview.buffer];

    size_t offset = bufferview.byteOffset + accessor.byteOffset;
    size_t count = accessor.count;

    out_attr.indices.resize(count); 
    std::memcpy(out_attr.indices.data(), &buffer.data[offset], count * sizeof(glm::uint16));
  }

  return true; 
}
