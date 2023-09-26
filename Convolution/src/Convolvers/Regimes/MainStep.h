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
#include "../Allocators/AllocatorMainStep.h"

namespace Convolution
{
	template<size_t WellFluxCount>
	struct MainStepWell :
		public KernelMainStep,
		public FluxMainStep
	{
		using Kernel = KernelMainStep;
		using Flux = FluxMainStep;

		using WellFluxMulti =
			CommonFluxMulti<
				typename Flux, 
				BaseWellFluxMainStep,
				Grids::AllNodeGroups::nodes_id::size
			>;

		MainStepWell(
			size_t spatial_size,
			size_t frame_temporal_size,
			size_t M,
			size_t small_step_nmbr,
			size_t main_step_nmbr) :
			KernelMainStep
		{
			MemoryDesc{spatial_size,
			frame_temporal_size},
			M, 
			small_step_nmbr,
			main_step_nmbr
		},
			FluxMainStep
		{
			MemoryDesc{spatial_size,
			main_step_nmbr},
			frame_temporal_size,
			small_step_nmbr
		}
		{}
	};

	template<size_t WellFluxCount>
	struct MainStepFrac :
		public MainStepWell<WellFluxCount>
	{
		template<typename Allocator_t>
		using CommonFluxMulti_Alloc = 
			CommonFluxMulti<Allocator_t,
			BaseFracFluxMainStep,
			Grids::AllNodeGroups::nodes_id::size>;

		using FracFluxMultiContainer = 
			FracturesFluxContainer_t<
			typename Flux, CommonFluxMulti_Alloc>;

		std::vector<typename MainStepWell::Kernel> 
			fracKernelRegime;
		std::vector<typename MainStepWell::Flux> 
			fracFluxRegime;

		/**
		 * 
		 * @param fracNy vector of number of y-nodes 
		 * along fractures
		 * 
		 * @param well_spatial_size nmbr of nodes
		 * along the well
		 * 
		 * @pram temporal_size nmbr of history entries.
		 * Amount of memory that sould be allocated 
		 * for fluxes.
		 */
		MainStepFrac(
			size_t well_spatial_size,
			size_t frame_temporal_size,
			size_t M,
			size_t small_step_nmbr,
			size_t main_step_nmbr,
			const std::vector<size_t>& fracNy) :

			MainStepWell{
			well_spatial_size,
			frame_temporal_size,
			M,
			small_step_nmbr,
			main_step_nmbr }
		{
			size_t frac_nmbr = fracNy.size();

			fracKernelRegime.reserve(frac_nmbr);
			fracFluxRegime.reserve(frac_nmbr);

			for (size_t frac_id = 0; 
				frac_id < frac_nmbr; ++frac_id
				)
			{
				fracKernelRegime.emplace_back(
					fracNy[frac_id],
					frame_temporal_size,
					M,
					small_step_nmbr,
					main_step_nmbr);
				fracFluxRegime.emplace_back(
					fracNy[frac_id], main_step_nmbr,
					frame_temporal_size,
					small_step_nmbr);
			}
		}
	};


	struct TimePolicyMainStep :
		public TimePolicyConstStep
	{
		using TimePolicyConstStep::TimePolicyConstStep;
	};

	template<size_t WellFluxCount>
	struct MainStepPolicy :
		public MainStepFrac<WellFluxCount>,
		public TimePolicyMainStep
	{
		using TimePolicy = TimePolicyMainStep;

		MainStepPolicy(const MainStepFrac& mainStep,
			const TimePolicyMainStep& timePolicy) noexcept :
			MainStepFrac{ mainStep },
			TimePolicyMainStep{ timePolicy }
		{}
	};

	template<size_t WellFluxCount>
	using MainStep = MainStepPolicy<WellFluxCount>;
} // Convolution
