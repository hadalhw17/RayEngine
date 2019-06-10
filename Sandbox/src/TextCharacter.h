#pragma once
#include <Character.h>
class TextCharacter :
	public RCharacter
{
public:
	TextCharacter() {}
};

namespace meta {

	template <>
	inline auto registerMembers<TextCharacter>()
	{
		return members(
			member("components", &TextCharacter::components),
			member("object_properties", &TextCharacter::object_properties),
			member("collision_box", &TextCharacter::collision_box),
			member("camera", &TextCharacter::camera)
		);
	}

} // end of namespace meta