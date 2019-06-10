#pragma once

#include "MeshObject.h"

namespace RayEngine
{
	class AFloor :
		public RMeshObject
	{
	public:
		AFloor();
		~AFloor();

		virtual void tick(float delta_time) override;
	};

}