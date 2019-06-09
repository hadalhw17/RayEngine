#pragma once
#include "MeshObject.h"

namespace RayEngine
{
	class AGlass :
		public RMeshObject
	{
	public:
		AGlass();
		~AGlass();

		virtual void tick(float delta_time);
	};

}