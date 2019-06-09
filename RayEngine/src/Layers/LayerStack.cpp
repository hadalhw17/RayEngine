#include "LayerStack.h"

RayEngine::RLayerStack::RLayerStack()
{
	m_layer_insert = m_layers.begin();
}

RayEngine::RLayerStack::~RLayerStack()
{
	for (auto layer : m_layers)
		delete layer;
}

void RayEngine::RLayerStack::push_layer(RLayer* layer)
{
	m_layer_insert = m_layers.emplace(m_layer_insert, layer);
}

void RayEngine::RLayerStack::push_overlay(RLayer* layer)
{
	m_layers.emplace_back(layer);
}

void RayEngine::RLayerStack::pop_layer(RLayer* layer)
{
	auto it = std::find(m_layers.begin(), m_layers.end(), layer);
	if (it != m_layers.end())
	{
		m_layers.erase(it);
		m_layer_insert--;
	}
}

void RayEngine::RLayerStack::pop_overlay(RLayer* layer)
{
	auto it = std::find(m_layers.begin(), m_layers.end(), layer);
	if (it != m_layers.end())
	{
		m_layers.erase(it);
	}
}
