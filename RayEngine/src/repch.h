#pragma once

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <functional>
#include <utility>
#include <cmath>
#include <limits>
#include <random>

#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "RayEngine/RayEngine.h"
#include <thread>
#include <fstream>

#include "MouseButtonCodes.h"
#include "KeyCodes.h"
#include "RayEngine/RayEngine.h"

#include "Events/Event.h"

#include "Camera.h"
#include "MovableCamera.h"
#include "KDTree.h"

#include <stdio.h>

#include <vector_types.h>
#include "helper_math.h"



#ifdef RE_PLATFORM_WINDOWS
	#include <Windows.h>
#endif // RE_PLATFORM_WINDOWS
