/*****************************************************************//**
 * \file   ConstStepAllocator.h
 * \brief  The file contains allocator definitions for 
 * the ConstStep regime simulations.
 * 
 * It specifies data access (struct ConstStep)
 * on read (OnGet...) and write (OnPush...)
 * for kernels (struct KernelConstStep)
 * and fluxes (struct FluxConstStep).
 * 
 * \author artur.salamatin
 * \date   June 2023
 *********************************************************************/

#pragma once
#include "../ConvolutionDefines.h"

namespace Convolution
{
	/**
	 * @brief
	 * Concrete descriptor of data
	 * that is going to be used for
	 * convolution at a next time moment.
	 *
	 * It is for KERNEL data
	 * at ConstStep regime.
	 *
	 * The indices must be within the allocated memory.
	 *
	 * @param its_index_end end position
	 * of pushed data within the memory
	 */
	struct OnGetKernelConstStep : public GetDesc
	{
		OnGetKernelConstStep(
			const GetDesc& memoryDesc) noexcept :
			GetDesc{ memoryDesc },
			its_index_end{ 0 }
		{}

		void on_extract() noexcept
		{
			// the data is going to be pulled from
			// Container.
			// This fact must be fixed
			// by updating descriptors before
			// the data is pulled.
			// So, now we get a proper
			// set of Kernel coefficients, again
			if (!is_external_boundary_time())
			{
				// by this if-statement we take into account
				// the external boundary
				++MemoryDesc::cur_temporal_window;
				its_index_end += MemoryDesc::spatial_size();
			}
		}

		// for kernel, in ConstStep regime: 
		// start frame position is always 
		// at the same place, 
		// begin()
		constexpr size_t idx_begin() const noexcept {
			return 0ull;
		}

		size_t idx_end() const noexcept {
			return its_index_end;
		}
	protected:
		size_t its_index_end;
		bool is_external_boundary_time() const
		{
			return idx_end() == MemoryDesc::allocated_memory();
		}
	};

	/**
	 * @brief
	 * Concrete descriptor of data
	 * that is going to be used for
	 * convolution at a next time moment.
	 *
	 * It is for FLUX data (well or fracture)
	 * at ConstStep regime.
	 *
	 * The indices must be within the allocated memory.
	 *
	 * @param its_index_begin start position
	 * of extracted data within the memory
	 *
	 * @param its_index_end end position
	 * of extracted data within the memory
	 * 
	 * @param frame_temporal_size mas nmbr 
	 * of time moments that participate
	 */
	struct OnGetFluxConstStep : public GetDesc
	{
		OnGetFluxConstStep(const OnGetFluxConstStep&) = default;

		OnGetFluxConstStep(
			const GetDesc&
			memoryDesc,
			size_t frame_temporal_size) noexcept :
			GetDesc{ memoryDesc },
			its_index_begin{ GetDesc::allocated_memory() }, // initially begin and end point outside the allocated memory
			its_index_end{ GetDesc::allocated_memory() }, // as there is no data yet
			frame_temporal_size{ frame_temporal_size }
		{}

		void on_extract() noexcept
		{
			// here we DO CARE about 
			// the external boundary
			if (is_external_boundary_time())
			{
				// the external boundary is reached
				// we can forget old source-terms
				its_index_end -= GetDesc::spatial_size();
			}
			else
			{// the external boundary is not reached yet
				++GetDesc::cur_temporal_window;
			}
			its_index_begin -= GetDesc::spatial_size();
		}

		// for kernel, in ConstStep regime: 
		// frame position is always at the same place, 
		// begin()
		size_t idx_begin() const noexcept {
			return its_index_begin;
		}

		size_t idx_end() const noexcept {
			return its_index_end;
		}
	protected:
		size_t its_index_begin;
		size_t its_index_end;
		const size_t frame_temporal_size;

		bool is_external_boundary_time() const
		{
			return 
				GetDesc::cur_temporal_window == 
				frame_temporal_size;
		}
	};

	struct OnPushKernelConstStep : public PushDesc
	{
		OnPushKernelConstStep(
			const PushDesc&
			positionDesc) noexcept :
			PushDesc{ positionDesc },
			its_index_end{ 0 } // there is no data yet
		{}

		void on_push() noexcept
		{
			++PushDesc::cur_temporal_window;
			its_index_end += PushDesc::spatial_size();
#ifdef PUSHER_ADVANCE_FLAG
			PushDesc::need_advance = false; // recently added data is fixed within the container, one can safely convolve
#endif
		}

		// for kernel, in ConstStep regime: 
		// frame position is always at the same place, 
		// begin()
		constexpr size_t idx_begin() const noexcept {
			return 0ull;
		}

		size_t idx_end() const noexcept {
			return its_index_end;
		}

	protected:
		size_t its_index_end;
	};

	struct OnPushFluxConstStep : public PushDesc
	{
		OnPushFluxConstStep(
			const PushDesc&
			memoryDesc) noexcept :
			PushDesc{ memoryDesc },
			// initially begin and end 
			// point outside the allocated memory
			its_index_begin{ allocated_memory() }
		{}

		void on_push() noexcept
		{
			// here we do not care about 
			// the external boundary

			// a new set of coefficients 
			// (per time moment) is added
			++PushDesc::cur_temporal_window;
			// in the const step regime
			// for Flux data
			// we only decrement the 
			// begin-index of the memory frame
			its_index_begin -= PushDesc::spatial_size();
#ifdef PUSHER_ADVANCE_FLAG
			// since the data is pushed safely,
			// it can be used later
			PushDesc::need_advance = false;
#endif
		}

		// for kernel, in ConstStep regime: 
		// frame position is always at the same place, 
		// begin()
		size_t idx_begin() const noexcept {
			return its_index_begin;
		}

		size_t idx_end() const noexcept {
			return MemoryDesc::allocated_memory();
		}
	protected:
		size_t its_index_begin;
	};

	struct KernelConstStep 
		:
		public Allocator
		<
		OnPushKernelConstStep,
		OnGetKernelConstStep
		>
	{
		KernelConstStep(const MemoryDesc& memoryDesc) :
			Allocator<OnPushKernelConstStep,
			OnGetKernelConstStep>{ 
			OnPushKernelConstStep{memoryDesc}, 
			OnGetKernelConstStep{memoryDesc} }
		{}

		KernelConstStep(
			size_t spatial_size,
			size_t frame_temporal_size) :
			KernelConstStep{ 
			MemoryDesc{spatial_size, frame_temporal_size} }
		{}
	};

	struct FluxConstStep 
		:
		public Allocator
		<
		OnPushFluxConstStep,
		OnGetFluxConstStep
		>
	{
		FluxConstStep(
			const MemoryDesc& memoryDesc, 
			size_t frame_temporal_size) 
			:
			Allocator<OnPushFluxConstStep,
			OnGetFluxConstStep>{ 
			OnPushFluxConstStep{memoryDesc},
			OnGetFluxConstStep{memoryDesc, frame_temporal_size } }
		{}

		FluxConstStep(
			size_t spatial_size,
			size_t temporal_size,
			size_t frame_temporal_size) :
			FluxConstStep{ 
				MemoryDesc{spatial_size, temporal_size},
				frame_temporal_size }
		{}
	};
} // Convolution

