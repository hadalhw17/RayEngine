#include "Color.h"



RColor::RColor()
{
	Red = 0.5, Green = 0.5, Blue = 0.5;
}

RColor::RColor(float r, float g, float b, float s)
{
	Red = r, Green = g, Blue = b, Special = s;
}

float RColor::Brightness()
{
	return(Red + Green + Blue) / 3;
}

RColor RColor::ColorScalar(float scalar)
{
	return RColor(Red*scalar, Green*scalar, Blue*scalar, Special);
}

RColor RColor::ColorAdd(RColor Color)
{
	return RColor(Red + Color.GetColorRed(), Green + Color.GetColorGreen(), Blue + Color.GetColorBlue(), Special);
}

RColor RColor::ColorMultiply(RColor Color)
{
	return RColor(Red*Color.GetColorRed(), Green*Color.GetColorGreen(), Blue*Color.GetColorBlue(), Special);
}

RColor RColor::ColorAverage(RColor Color)
{
	return RColor((Red + Color.GetColorRed()) / 2, (Green + Color.GetColorGreen()) / 2, (Blue + Color.GetColorBlue()) / 2, Special);
}

RColor RColor::Clip()
{
	float alllight = Red + Green + Blue;
	float excesslight = alllight - 3;
	if (excesslight > 0) {
		Red = Red + excesslight * (Red / alllight);
		Green = Green + excesslight * (Green / alllight);
		Blue = Blue + excesslight * (Blue / alllight);
	}
	if (Red > 1) { Red = 1; }
	if (Green > 1) { Green = 1; }
	if (Blue > 1) { Blue = 1; }
	if (Red < 0) { Red = 0; }
	if (Green < 0) { Green = 0; }
	if (Blue < 0) { Blue = 0; }

	return RColor(Red, Green, Blue, Special);
}
