// Define these only in *one* .cc file.
#include "meshio.h"

#include <tiny_gltf.h>
#include <glm/gtc/type_ptr.hpp>

#include <cstdio>

template <typename T>
void castToIntArray(const std::vector<unsigned char>& buffer, size_t offset, size_t count, std::vector<glm::uint32>& result) {
  result.clear();
  result.reserve(count); 

  for (size_t i = 0; i < count; ++i) {
    T value; 
    std::memcpy(&value, &buffer[offset + i * sizeof(T)], sizeof(T));
    result.push_back(static_cast<glm::uint32>(value));
  }
}

bool loadGLTFImageBuffer(meshio::ImageData& outImageData, const tinygltf::Image& image) {
  // right now, lets assume RGBA (4-channel) color, 8-bit depth.
  // If something else comes along we'll deal with it. 
  if (image.component != 4 ||
    image.bits != 8 ||
    image.pixel_type != TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
    std::fprintf(stderr, "Texture could not be loaded: Unsupported image format.\n");
    return false;
  }

  // also image could be in buffer view, I'm not sure how 
  // tinygltf handles this, so yell at me so I can investigate. 
  if (image.bufferView != -1) {
    std::fprintf(stderr, "Texture in bin data, not supported?\n");
    return false;
  }

  outImageData.width = image.width;
  outImageData.height = image.height;

  size_t numPixels = static_cast<size_t>(image.width) * image.height;
  std::vector<glm::uint8> imageBuffer;
  imageBuffer.resize(numPixels * 4);

  std::memcpy(imageBuffer.data(), image.image.data(), numPixels * 4 * sizeof(glm::uint8));

  for (size_t i = 0; i < numPixels; ++i) {
    size_t idx = i * 4;
    glm::vec4 color;
    color.r = imageBuffer[idx] / 255.f;
    color.g = imageBuffer[idx + 1] / 255.f;
    color.b = imageBuffer[idx + 2] / 255.f;
    color.a = imageBuffer[idx + 3] / 255.f;
    outImageData.buffer.push_back(color);
  }

  return true; 
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

  // TODO: node may have transforms, we should factor that in?

  // primitives contain index for attribute buffers, materials, draw mode
  tinygltf::Primitive& prim = model.meshes[0].primitives[0];

  // must be triangles
  if (prim.mode != TINYGLTF_MODE_TRIANGLES) {
    return false; 
  }

  int posIdx, norIdx, texIdx; 
  {
    auto it = prim.attributes.find("POSITION");
    posIdx = it == prim.attributes.end() ? -1 : it->second;

    it = prim.attributes.find("NORMAL");
    norIdx = it == prim.attributes.end() ? -1 : it->second;

    // assumes only one set of texture coordinates
    it = prim.attributes.find("TEXCOORD_0"); 
    texIdx = it == prim.attributes.end() ? -1 : it->second; 
  }

  // must at least have positions / index array!
  if (posIdx == -1 || prim.indices == -1) {
    return false; 
  } 

  // fill index array
  {
    tinygltf::Accessor& accessor = model.accessors[prim.indices];
    tinygltf::BufferView& bufferview = model.bufferViews.at(accessor.bufferView);   // buffer views index is optional

    // indices must be scalar
    if (accessor.type != TINYGLTF_TYPE_SCALAR) {
      return false;
    }

    tinygltf::Buffer& buffer = model.buffers[bufferview.buffer];

    size_t offset = bufferview.byteOffset + accessor.byteOffset;
    size_t count = accessor.count;

    switch (accessor.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      castToIntArray<glm::uint8>(buffer.data, offset, count, out_attr.indices);
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      castToIntArray<glm::uint16>(buffer.data, offset, count, out_attr.indices);
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
      castToIntArray<glm::uint32>(buffer.data, offset, count, out_attr.indices);
      break;
    default:
      return false;
    }
  }

  // fill positions array
  {
    tinygltf::Accessor& accessor = model.accessors[posIdx];
    tinygltf::BufferView& bufferview = model.bufferViews.at(accessor.bufferView);   // buffer views index is optional
    
    // We must have vec3's of floats
    if (accessor.type != TINYGLTF_TYPE_VEC3 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
      std::fprintf(stderr, "Position data in unsupported format.\n");
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
      std::fprintf(stderr, "Normal data in unsupported format.\n");
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

  // fill tex coords array
  if (texIdx != -1) {
    tinygltf::Accessor& accessor = model.accessors[texIdx];
    tinygltf::BufferView& bufferview = model.bufferViews.at(accessor.bufferView);   // buffer views index is optional

    if (accessor.type != TINYGLTF_TYPE_VEC2 || accessor.componentType != TINYGLTF_COMPONENT_TYPE_FLOAT) {
      std::fprintf(stderr, "Texture coordinate data in unsupported format.\n");
      return false; 
    }

    tinygltf::Buffer& buffer = model.buffers[bufferview.buffer];

    size_t offset = bufferview.byteOffset + accessor.byteOffset;
    size_t count = accessor.count;

    std::vector<glm::float32> tex; 
    tex.resize(count * 2);    // vec2

    std::memcpy(tex.data(), &buffer.data[offset], count * sizeof(glm::float32) * 2); 

    for (size_t i = 0; i < count; ++i) {
      out_attr.texcoords.emplace_back(glm::make_vec2(&tex[i * 2]));
    }
  }

  // we look for textures through the model's material
  if (prim.material != -1 && !model.images.empty()) {
    tinygltf::Material material = model.materials[prim.material]; 

    int baseColorIdx = material.pbrMetallicRoughness.baseColorTexture.index; 
    int normalIdx = material.normalTexture.index; 

    if (baseColorIdx != -1) {
      if (!loadGLTFImageBuffer(out_attr.textureAlbedo, model.images[baseColorIdx])) {
        return false; 
      }
    }

    if (normalIdx != -1) {
      if (!loadGLTFImageBuffer(out_attr.textureNormal, model.images[normalIdx])) {
        return false; 
      }
    }
  }

  return true; 
}
