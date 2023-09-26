#include "Test1.h"

#include "Convolvers/ConvolutionDefines.h"
#include "Convolvers/Allocators/AllocatorConstStep.h"
#include "Convolvers/Kernels/BaseKernel.h"

#include "../Factory/ClassFactory.h"
#include "../Printers/Printers.h"

namespace Tests
{
	bool test_memDesc()
	{
		size_t source_count{ 100 };
		size_t time_intervals_count{ 20 };

		ClassFactory factory{ source_count, time_intervals_count };

		Convolution::MemoryDesc memDesc
		{
			factory.create_memoryDesc()
		};

		std::cout << memDesc << std::endl;

		return source_count == memDesc.spatial_size() &&
			time_intervals_count == memDesc.temporal_size() &&
			source_count * time_intervals_count == memDesc.allocated_memory();
	}

	bool test_onGetKernelConstStep()
	{
		size_t source_count{ 100 };
		size_t time_intervals_count{ 20 };

		ClassFactory factory{ source_count, time_intervals_count };

		auto getKernelConstStep{ 
			factory.create_allocatorBlockConstStep<Convolution::OnGetKernelConstStep>()};

		std::cout << getKernelConstStep << std::endl;

		return 0ull == getKernelConstStep.idx_begin() &&
			0ull == getKernelConstStep.idx_end();
	}

	bool test_onGetFluxConstStep()
	{
		size_t source_count{ 100 };
		size_t time_intervals_count{ 20 };
		size_t frame_temporal_size{ 10 };

		ClassFactory factory{ source_count, time_intervals_count, frame_temporal_size };

		Convolution::OnGetFluxConstStep getFluxConstStep{
			factory.create_onGetFluxConstStep() };

		std::cout << getFluxConstStep << std::endl;

		return getFluxConstStep.allocated_memory() == getFluxConstStep.idx_begin() &&
			getFluxConstStep.allocated_memory() == getFluxConstStep.idx_end();
	}

	bool test_kernelConstStep()
	{
		size_t source_count{ 100 };
	//	size_t time_intervals_count{ 20 };
		size_t frame_temporal_size{ 10 };

		ClassFactory factory{ source_count, frame_temporal_size };

		Convolution::KernelConstStep kernel{
			factory.create_allocatorBlockConstStep
			<
				Convolution::KernelConstStep
			>
			() 
		};

		std::cout << kernel << std::endl;

		return false;
	}
	

	bool test_baseKernel_constStep()
	{
		size_t rows_count{ 300'000ull };
		size_t source_count{ 100 };
		//	size_t time_intervals_count{ 20 };
		size_t frame_temporal_size{ 10 };

		ClassFactory factory{ source_count, frame_temporal_size };

		Convolution::KernelConstStep allocator{
			factory.create_allocatorBlockConstStep
			<
				Convolution::KernelConstStep
			>
			()
		};

		Convolution::BaseKernel<Convolution::KernelConstStep>
			well_kernel{rows_count, allocator };

		std::cout << well_kernel << std::endl;

		return false;
	}
}
