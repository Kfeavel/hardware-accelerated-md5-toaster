#pragma once
#include "cuda_runtime.h"
#include "main.h"


namespace md5scope {
	void md5_calculate(struct cuda_device *device);
}