/*****************************************************************//**
 * \file   AllocatorMainStep.h
 * \brief  The file contains allocator definitions for
 * the MainStep calculations within the VarStep regime
 * 
 * It specifies data access (struct MainStep)
 * on read (OnGet...) and write (OnPush...)
 * for kernels (struct KernelMainStep)
 * and fluxes (struct FluxMainStep)
 * 
 * \author artur.salamatin
 * \date   June 2023
 *********************************************************************/

#pragma once
#include "AllocatorConstStep.h"

namespace Convolution
{
	/**
	 * @brief
	 * Concrete descriptor of data
	 * that is going to be used for
	 * convolution at a next time moment.
	 *
	 * It is for KERNEL data
	 * and MainStep terms
	 * at VarStep regime.
	 *
	 * The indices must be within the allocated memory.
	 *
	 * @param its_index_end end position
	 * of pushed data within the memory
	 * @param M number of main steps of the second part of history
	 */
	struct OnGetKernelMainStep :
		public OnGetKernelConstStep
	{
		OnGetKernelMainStep(
			const GetDesc& memoryDesc,
			size_t M,
			size_t small_step_nmbr_,
			size_t main_step_nmbr_) noexcept :
			OnGetKernelConstStep{ memoryDesc },
			its_index_begin{ 0ull },
			small_step_nmbr{ small_step_nmbr_ },
			small_step_counter{ 0ull },
			M{ M },
			main_step_nmbr{main_step_nmbr_}, 
			main_step_counter{0ull}
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
			// 
			// set these class fields:
			// cur_temporal_window
			// its_idx_begin
			// its_index_end

			// check whether we include the split steps
			if (is_first_history_period())
			{
				// we are still within the first part
				// of the history.
				// It essentially follows the ConstStep 
				// regime
				OnGetKernelConstStep::on_extract();
				++main_step_counter;
			}
			else
			{
				// we get data to convolve
				// within the small step of the history
				if (small_step_counter == 0)
				{
					// We are at the very first 
					// small step
					// within a main step.
					// Move the begin index.
					its_index_begin += 
						MemoryDesc::spatial_size();
					if (
						!OnGetKernelConstStep::
						is_external_boundary_time())
					{
						// by this if-statement we 
						// take into account
						// the external boundary,
						// if it is encountered while we are 
						// in the SECOND part of history
						OnGetKernelConstStep::
							its_index_end 
							+= MemoryDesc::spatial_size();
					}
					// remains unchanged
					// MemoryDesc::cur_temporal_window;
				}
				++small_step_counter;
				// AAS: modulo /*остаток от деления*/
				small_step_counter %= small_step_nmbr;
			}
		}

		// for kernel, in ConstStep regime: 
		// start frame position is always 
		// at the same place, 
		// begin()
		size_t idx_begin() const {
			return its_index_begin;
		}
		size_t current_window_size() const {
			return OnGetKernelConstStep::idx_end() - 
				idx_begin();
		}

	protected:
		// once the second part of history is 
		// entered, the begin part of 
		size_t its_index_begin;
		// nmbr of split MainSteps
		const size_t M; 
		// nmbr of small steps per main step
		const size_t small_step_nmbr;
		// counter for small steps within the 
		// main(large) step
		size_t small_step_counter;

		const size_t main_step_nmbr;
		size_t main_step_counter;

	private:
		bool is_first_history_period() const
		{
			return main_step_counter < main_step_nmbr;
		}
	};

	/**
	 * @brief
	 * Concrete descriptor of data
	 * that is going to be used for
	 * convolution at a next time moment.
	 *
	 * It is for FLUX data (well or fracture)
	 * and MainStep terms
	 * at VarStep regime.
	 *
	 * The indices must be within the allocated memory.
	 *
	 * @param its_index_begin start position
	 * of extracted data within the memory
	 *
	 * @param its_index_end end position
	 * of extracted data within the memory
	 */
	struct OnGetFluxMainStep : 
		//public GetDesc
		public OnGetFluxConstStep
	{
		using OnGetFluxConstStep::OnGetFluxConstStep;

		void on_extract() noexcept
		{
			if (is_first_history_period())
			{
				// We are in a first part of history.
				// The behavior is the same as in
				// the ConstStep regime
				OnGetFluxConstStep::on_extract();
			}
			else
			{
				// idx_begin() == 0ull

				// In the second part of history
				// the behavior is different.
				// 
				if (OnGetFluxConstStep::
					is_external_boundary_time())
				{
					// the external boundary is reached
					// we can forget old source-terms
					if(idx_end() > idx_begin())
						// We can ony firget data
						// until we still have any
						// unforgotten data
						its_index_end -= 
						GetDesc::spatial_size();
				}
				else
				{// the external boundary is not reached yet
					++GetDesc::cur_temporal_window;
				}
			}
		}

	protected:
		bool is_first_history_period() noexcept
		{
			// the container only stores 
			// fluxes for the first period of history
			return idx_begin() > 0ull;
		}
	};

	/**
	 * \brief It is essentially the same 
	 * imlementation as for the ConstStep.
	 */
	struct OnPushKernelMainStep : 
		public OnPushKernelConstStep
	{
		using OnPushKernelConstStep::OnPushKernelConstStep;
	};

	struct OnPushFluxMainStep : public PushDesc
	{
		OnPushFluxMainStep(
			const PushDesc&
			memoryDesc) noexcept :
			PushDesc{ memoryDesc },
			// initially begin and end 
			// point outside the allocated memory
			its_index_begin{ allocated_memory() }
		{}

		void on_push() noexcept
		{
			// Here we do not care about 
			// the external boundary.
			// It is checked outside.

			// a new set of coefficients 
			// (per time moment) is added
			++PushDesc::cur_temporal_window;
			// in the const step regime
			// for Flux data
			// we only decrement the 
			// begin-index of the memory frame
			its_index_begin -= PushDesc::spatial_size();
			// since the data is pushed safely,
			// it can be used later
#ifdef PUSHER_ADVANCE_FLAG			
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

	struct KernelMainStep :
		public Allocator
		<
		OnPushKernelMainStep,
		OnGetKernelMainStep
		>
	{
		KernelMainStep(
			const MemoryDesc& memoryDesc, 
			size_t M,
			size_t small_step_nmbr,
			size_t main_step_nmbr) :
			Allocator<OnPushKernelMainStep,
			OnGetKernelMainStep>{
			OnPushKernelMainStep{memoryDesc},
			OnGetKernelMainStep{
					memoryDesc, 
					M, small_step_nmbr,
					main_step_nmbr} }
		{}

		KernelMainStep(
			size_t spatial_size,
			size_t frame_temporal_size,
			size_t M,
			size_t small_step_nmbr,
			size_t main_step_nmbr) :
			KernelMainStep{
			MemoryDesc{spatial_size, frame_temporal_size},
			M, small_step_nmbr, main_step_nmbr }
		{}
	};

	struct FluxMainStep
		:
		public Allocator
		<
		OnPushFluxMainStep,
		OnGetFluxMainStep
		>
	{
		FluxMainStep(
			const MemoryDesc& memoryDesc,
			size_t frame_temporal_size,
			size_t small_step_nmbr)
			:
			Allocator<OnPushFluxMainStep,
			OnGetFluxMainStep>
			{
				OnPushFluxMainStep{
						memoryDesc
				},
				OnGetFluxMainStep{
						memoryDesc,
						frame_temporal_size
				} 
			},
			small_step_nmbr{ small_step_nmbr },
			main_step_nmbr{ memoryDesc.temporal_size()}
		{}

		FluxMainStep(
			size_t spatial_size,
			size_t main_step_nmbr,
			size_t frame_temporal_size,
			size_t small_step_nmbr) :
			FluxMainStep{
				MemoryDesc{spatial_size, main_step_nmbr},
				frame_temporal_size,
				small_step_nmbr}
		{}

		// additional memory, purely related 
		// to MainStep regime
		const size_t small_step_nmbr;
		const size_t main_step_nmbr;
	};
} // Convolution
