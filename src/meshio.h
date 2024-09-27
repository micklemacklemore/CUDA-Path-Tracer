#include "tiny_gltf.h"
#include "glm/glm.hpp"


#include <string>

struct MeshAttributes {
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::vec2> texcoords;
  std::vector<glm::uint32> indices;
  
  // texture info
  struct ImageData {
    std::vector<glm::vec4> buffer;
    glm::uint32 width, height;
  } image;
};


namespace meshio {
		bool loadMesh(std::string filename, MeshAttributes& mesh); 
}