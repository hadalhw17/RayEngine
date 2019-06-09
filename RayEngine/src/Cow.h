#pragma once
#include "MeshObject.h"

namespace RayEngine
{
	class ACow :
		public RMeshObject
	{
	public:
		ACow();
		~ACow();

		virtual void tick(float delta_time);
	};

}