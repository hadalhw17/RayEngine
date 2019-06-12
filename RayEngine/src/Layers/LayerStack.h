#pragma once


#include "Layer.h"
#include <vector>

namespace RayEngine
{
	class RAY_ENGINE_API RLayerStack
	{
	public:
		RLayerStack();
		~RLayerStack();

		void push_layer(RLayer* layer);
		void push_overlay(RLayer* layer);
		void pop_layer(RLayer* layer);
		void pop_overlay(RLayer* layer);

		std::vector<RLayer*>::iterator begin() { return m_layers.begin(); }
		std::vector<RLayer*>::iterator end() { return m_layers.end(); }

		std::vector<RLayer*> m_layers;
		std::vector<RLayer*>::iterator m_layer_insert;
	private:
	};

}

#include <Meta.h>
namespace meta {

	template <>
	inline auto registerMembers<RayEngine::RLayerStack>()
	{
		return members(
			member("m_layers", &RayEngine::RLayerStack::m_layers),
			member("m_layer_insert", &RayEngine::RLayerStack::m_layer_insert)
		);
	}
} // end of namespace meta