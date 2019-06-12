#pragma once
#include <cmath> 
#include <cstdio> 
#include <random> 
#include <functional> 
#include <fstream> 
#include <algorithm> 
#include "RayEngine/RayEngine.h"
#include "helper_math.h"

namespace RayEngine
{
	class RAY_ENGINE_API RPerlinNoise
	{
	public:
		RPerlinNoise(const unsigned& seed = 2031)
		{
			std::mt19937 generator(seed);
			std::uniform_real_distribution<float> distribution;
			auto dice = std::bind(distribution, generator);
			for (unsigned i = 0; i < tableSize; ++i) {
#if 0 
				// bad
				float gradientLen2;
				do {
					gradients[i] = Vec3f(2 * dice() - 1, 2 * dice() - 1, 2 * dice() - 1);
					gradientLen2 = gradients[i].length2();
				} while (gradientLen2 > 1);
				gradients[i].normalize();
#else 
				// better
				double theta = acos(2 * dice() - 1);
				double phi = 2.f * dice() * 3.14159265358979323846;

				double x = cos(phi) * sin(theta);
				double y = sin(phi) * sin(theta);
				double z = cos(theta);
				gradients[i] = make_float3(x, y, z);
#endif 
				permutationTable[i] = i;
			}

			std::uniform_int_distribution<unsigned> distributionInt;
			auto diceInt = std::bind(distributionInt, generator);
			// create permutation table
			for (unsigned i = 0; i < tableSize; ++i)
				std::swap(permutationTable[i], permutationTable[diceInt() & tableSizeMask]);
			// extend the permutation table in the index range [256:512]
			for (unsigned i = 0; i < tableSize; ++i) {
				permutationTable[tableSize + i] = permutationTable[i];
			}
		}
		virtual ~RPerlinNoise()
		{
		}

		static const unsigned tableSize = 256;
		static const unsigned tableSizeMask = tableSize - 1;
		float3 gradients[tableSize];
		unsigned permutationTable[tableSize * 2];
	};
}

