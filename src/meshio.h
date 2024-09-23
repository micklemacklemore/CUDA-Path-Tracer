#include "tiny_gltf.h"
#include "glm/glm.hpp"


#include <string>

struct MeshAttributes {
  std::vector<glm::vec3> positions;
  std::vector<glm::vec3> normals;
  std::vector<glm::uint16> indices;
};


namespace meshio {
		bool loadMesh(std::string filename, MeshAttributes& mesh); 
}