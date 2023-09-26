#include "ClassFactory.h"

ClassFactory::ClassFactory(
	size_t source_count,
	size_t time_intervals_count,
	size_t frame_temporal_size) :
	source_count{source_count},
	time_intervals_count{time_intervals_count},
	frame_temporal_size{ frame_temporal_size }
{ }

Convolution::MemoryDesc ClassFactory::create_memoryDesc()
{
	return Convolution::MemoryDesc{ source_count, time_intervals_count };
}

Convolution::OnGetFluxConstStep ClassFactory::create_onGetFluxConstStep()
{
	Convolution::MemoryDesc memDesc{ source_count, time_intervals_count };

	return Convolution::OnGetFluxConstStep{ memDesc, frame_temporal_size };
}
