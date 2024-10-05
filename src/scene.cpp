#include "scene.h"
#include "meshio.h"

#include "json.hpp"

#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/common.hpp>
#include <unordered_map>


using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}


void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);

    std::string jsonDirPath; 

    if (jsonName.rfind("\\") != string::npos) {
      jsonDirPath = jsonName.substr(0, jsonName.rfind("\\") + 1);
    } 
    else if (jsonName.rfind("/") != string::npos) {
      jsonDirPath = jsonName.substr(0, jsonName.rfind("/") + 1);
    }
    else {
      std::cerr << "loadFromJSON: failed to parse string." << std::endl;
      std::exit(-1); 
    }

    // Parse Materials

    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.f - p["ROUGHNESS"];   // this is probably wrong
        }
        else if (p["TYPE"] == "Transmission")
        {
          const auto& col = p["RGB"];
          newMaterial.color = glm::vec3(col[0], col[1], col[2]);
          newMaterial.hasTransmissive = 1.f; 
        }
        else if (p["TYPE"] == "MicrofacetReflection") {
          const auto& col = p["RGB"];
          const auto& rough = p["ROUGHNESS"]; 
          newMaterial.color = glm::vec3(col[0], col[1], col[2]);
          newMaterial.isMicrofacet = 1.f; 
          newMaterial.roughness = rough; 
        }
        MatNameToID[name] = materials.size();

        newMaterial.textureIdx.albedo = -1; 
        newMaterial.textureIdx.normal = -1; 

        materials.emplace_back(newMaterial);
    }
    
    // Parse Objects

    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
      const auto& type = p["TYPE"];
      if (type != "mesh") {
        Geom newGeom;
        if (type == "cube")
        {
          newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
          newGeom.type = SPHERE;
        }
        else {
          std::cerr << "loadFromJSON: do not recognise object type." << std::endl; 
          std::exit(-1); 
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
          newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
      }
      else {
        // MESH TYPE
        meshio::MeshAttributes mesh;
        const std::string filePath = p["PATH"]; 
        if (!meshio::loadMesh(jsonDirPath + filePath, mesh)) {
          std::cerr << "loadFromJSON: failed to load mesh: " << jsonDirPath + filePath << std::endl;
          std::exit(-1);
        }

        Geom geom; 
        geom.type = MESH; 

        // model transforms
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];

        glm::mat4 modelMat = utilityCore::buildTransformationMatrix(
          glm::vec3(trans[0], trans[1], trans[2]),
          glm::vec3(rotat[0], rotat[1], rotat[2]),
          glm::vec3(scale[0], scale[1], scale[2])
        ); 
        glm::mat4 modelMatInvTrans = glm::inverseTranspose(modelMat); 

        geom.materialid = MatNameToID[p["MATERIAL"]];;

        // fill textures if any
        if (mesh.textureAlbedo.exists()) {
          textures.emplace_back(std::move(mesh.textureAlbedo));
          materials[geom.materialid].textureIdx.albedo = textures.size() - 1;
        }

        if (mesh.textureNormal.exists()) {
          textures.emplace_back(std::move(mesh.textureNormal)); 
          materials[geom.materialid].textureIdx.normal = textures.size() - 1;
        }

        geom.triStart = tris.size(); 

        glm::vec3 min(FLT_MAX);
        glm::vec3 max(-FLT_MAX); 

        // fill positions
        for (size_t idx = 0; idx < mesh.indices.size(); idx += 3) {
          Triangle tri;

          tri.trianglePos[0] = mesh.positions[mesh.indices[idx]];
          tri.trianglePos[1] = mesh.positions[mesh.indices[idx + 1]];
          tri.trianglePos[2] = mesh.positions[mesh.indices[idx + 2]];

          if (!mesh.normals.empty()) {
            tri.triangleNor[0] = mesh.normals[mesh.indices[idx]];
            tri.triangleNor[1] = mesh.normals[mesh.indices[idx + 1]];
            tri.triangleNor[2] = mesh.normals[mesh.indices[idx + 2]];
          }

          if (!mesh.texcoords.empty()) {
            tri.triangleTex[0] = mesh.texcoords[mesh.indices[idx]];
            tri.triangleTex[1] = mesh.texcoords[mesh.indices[idx + 1]];
            tri.triangleTex[2] = mesh.texcoords[mesh.indices[idx + 2]];
          }

          tri.trianglePos[0] = glm::vec3(modelMat * glm::vec4(tri.trianglePos[0], 1.)); 
          tri.trianglePos[1] = glm::vec3(modelMat * glm::vec4(tri.trianglePos[1], 1.));
          tri.trianglePos[2] = glm::vec3(modelMat * glm::vec4(tri.trianglePos[2], 1.));

          tri.triangleNor[0] = glm::vec3(modelMatInvTrans * glm::vec4(tri.triangleNor[0], 1.)); 
          tri.triangleNor[1] = glm::vec3(modelMatInvTrans * glm::vec4(tri.triangleNor[1], 1.));
          tri.triangleNor[2] = glm::vec3(modelMatInvTrans * glm::vec4(tri.triangleNor[2], 1.));

          tris.push_back(tri); 

          // update min and max
          max = glm::max(max, tri.trianglePos[0]); 
          max = glm::max(max, tri.trianglePos[1]);
          max = glm::max(max, tri.trianglePos[2]);
          
          min = glm::min(min, tri.trianglePos[0]); 
          min = glm::min(min, tri.trianglePos[1]);
          min = glm::min(min, tri.trianglePos[2]);
        }

        geom.minBoundingBox = min; 
        geom.maxBoundingBox = max; 
        geom.triNum = tris.size() - geom.triStart; 

        geoms.push_back(geom); 
      }  
    }

    // Parse Camera 

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    // Calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    // Set final render resolution
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
