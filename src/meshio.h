#pragma once

#include "tiny_gltf.h"
#include "glm/glm.hpp"

#include <string>


namespace meshio {
  struct ImageData {
    std::vector<glm::vec4> buffer;
    glm::uint32 width, height;

    bool exists() {
      return !buffer.empty(); 
    }
  };

  struct MeshAttributes {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    std::vector<glm::uint32> indices;
    
    ImageData textureAlbedo; 
    ImageData textureNormal; 
    // emissive, roughness/metallic etc...
  };

	bool loadMesh(std::string filename, MeshAttributes& mesh); 
}