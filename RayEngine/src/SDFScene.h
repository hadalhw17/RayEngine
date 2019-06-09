#pragma once
#include "Scene.h"
#include "RayEngine/RayEngine.h"
#include "Grid.h"
#include "PerlinNoise.h"
#include "Character.h"
#include "TextureObject.h"
#include "Material.h"
#include <vector>


class RAY_ENGINE_API SDFScene :
	public RScene
{

public:
	SDFScene();
	SDFScene(RCharacter& character);
protected:
	virtual void build_scene() override;
protected:
	void init_cuda_res();
	RayEngine::RPerlinNoise noise;
	std::vector<VoxelMaterial> materials;
	Grid distance_field;
};

