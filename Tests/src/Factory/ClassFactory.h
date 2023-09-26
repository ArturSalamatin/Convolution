#pragma once

#include "Convolvers/ConvolutionDefines.h"
#include "Convolvers/Allocators/AllocatorConstStep.h"

struct ClassFactory
{
	ClassFactory(size_t source_count,
		size_t time_intervals_count,
		size_t frame_temporal_size = 0ull);

	Convolution::MemoryDesc create_memoryDesc();
	template<typename allocator_block_t>
	allocator_block_t create_allocatorBlockConstStep()
	{
		Convolution::MemoryDesc memDesc{ source_count, time_intervals_count };
		return allocator_block_t{ memDesc };
	}

	Convolution::OnGetFluxConstStep create_onGetFluxConstStep();

protected:
	size_t source_count;
	size_t time_intervals_count;
	size_t frame_temporal_size;
};

