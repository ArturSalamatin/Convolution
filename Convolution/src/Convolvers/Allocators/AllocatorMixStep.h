#pragma once

#include "AllocatorMainStep.h"

namespace Convolution
{
	struct OnGetKernelMixStep :
		public GetDesc
	{
		OnGetKernelMixStep(
			const GetDesc& memoryDesc,
			size_t small_step_nmbr_) noexcept :

			GetDesc{ memoryDesc },
			small_step_nmbr{ small_step_nmbr_},
			small_step_counter{ 0ull }
		{}

		void on_extract() noexcept
		{
		}

		// for kernel, in ConstStep regime: 
		// start frame position is always 
		// at the same place, 
		// begin()
		constexpr size_t idx_begin() const noexcept {
			return 0ull;
		}

		size_t idx_end() const noexcept {
			return GetDesc::allocated_memory();
		}

	protected:
		// nmbr of small steps per main step
		const size_t small_step_nmbr;
		size_t small_step_counter;
	};

	struct OnGetFluxMixStep :
		public GetDesc
	{
		OnGetFluxMixStep(
			const GetDesc& memoryDesc,
			size_t frame_temporal_size) noexcept :
			GetDesc{ memoryDesc },
			frame_temporal_size{ frame_temporal_size }
		{}

		// it does nothing
		void on_extract() noexcept
		{	
			if (!is_external_boundary_time())
			{// the external boundary is not reached yet
				++GetDesc::cur_temporal_window;
			}
		}

		constexpr size_t idx_begin() const noexcept
		{
			return 0ull;
		}

		size_t idx_end() const noexcept
		{
			return MemoryDesc::spatial_size();
		}

	protected:
		const size_t frame_temporal_size;

		bool is_external_boundary_time() const
		{
			return
				GetDesc::cur_temporal_window ==
				frame_temporal_size;
		}
	};

	struct OnPushKernelMixStep : public PushDesc
	{
		// We only keep single time segment in the memory,
		

		OnPushKernelMixStep(
			const PushDesc&
			positionDesc) noexcept :
			PushDesc{ positionDesc }
		{}

		void on_push() noexcept
		{
#ifdef PUSHER_ADVANCE_FLAG
			PushDesc::need_advance = false; // recently added data is fixed within the container, one can safely convolve
#endif
		}

		constexpr size_t idx_end() const noexcept {
			return 0ull;
		}
	};

	
	struct OnPushFluxMixStep : public OnPushFluxMainStep
	{
		using OnPushFluxMainStep::OnPushFluxMainStep;
	};

	struct KernelMixStep :
		public Allocator
		<
		OnPushKernelMixStep,
		OnGetKernelMixStep
		>
	{
		KernelMixStep(
			size_t spatial_size,
			size_t frame_temporal_size,
			size_t small_step_nmbr_per_main_step,
			size_t M
		) :
			KernelMixStep
			{
				MemoryDesc{spatial_size, frame_temporal_size},
				small_step_nmbr_per_main_step, M
			}
		{}

		/*size_t Pcur_cache_size() const { return M; }
		size_t Pprev_cache_size() const 
		{ 
			return small_step_nmbr_per_main_step;
		}*/

		const size_t M, small_step_nmbr_per_main_step;
	protected:
		KernelMixStep(
			const MemoryDesc& memoryDesc,
			size_t small_step_nmbr_per_main_step,
			size_t M
		) :
			Allocator<OnPushKernelMixStep,
			OnGetKernelMixStep>
			{
				OnPushKernelMixStep{memoryDesc},
				OnGetKernelMixStep{
					memoryDesc,
					small_step_nmbr_per_main_step}
			},
			M{ M },
			small_step_nmbr_per_main_step{ small_step_nmbr_per_main_step }
		{}

	};

	struct FluxMixStep
		:
		public Allocator
		<
		OnPushFluxMixStep,
		OnGetFluxMixStep
		>
	{
		FluxMixStep(
			size_t spatial_size,
			size_t frame_temporal_size) :
			FluxMixStep{
				MemoryDesc{spatial_size, 1ull},
				frame_temporal_size }
		{}

	protected:
		FluxMixStep(
			const MemoryDesc& memoryDesc,
			size_t frame_temporal_size)
			:
			Allocator<
			OnPushFluxMixStep,
			OnGetFluxMixStep
			>
		{
			OnPushFluxMixStep{
					memoryDesc
			},
			OnGetFluxMixStep{
					memoryDesc,
					frame_temporal_size
			}
		}
		{}
	};
} // Convolution
