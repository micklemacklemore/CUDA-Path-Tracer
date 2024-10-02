#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "meshio.h"

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(string filename);
    ~Scene() {}; 

    std::vector<Geom> geoms;
    std::vector<Triangle> tris; 
    std::vector<Material> materials;
    std::vector<meshio::ImageData> textures; 
    RenderState state;
};
