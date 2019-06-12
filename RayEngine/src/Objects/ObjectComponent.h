#pragma once
class RObjectComponent
{
public:
	RObjectComponent();
	~RObjectComponent();
};

#include <Meta.h>
namespace meta {

	template <>
	inline auto registerMembers<RObjectComponent>()
	{
		return members(
		);
	}

} // end of namespace meta