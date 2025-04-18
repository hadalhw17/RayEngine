#pragma once

#include <vector_types.h>
#include "Grid.h"


namespace RayEngine
{
	class RAY_ENGINE_API RChunk
	{
	public:
		RChunk();
		RChunk(Grid& _distance_field);
		~RChunk();

		inline const Grid& get_sdf() const { return distance_field; }
		inline const float3& get_location() const { return location; }

		void set_location(float3 _location) { location = _location; }
		void update_sdf_box(const float3& new_box_max);
	private:
		Grid distance_field;
		float3 location;
	};
}
