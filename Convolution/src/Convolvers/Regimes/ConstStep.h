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
#include "../Fluxes/WellFlux.h"
#include "../Fluxes/FracFlux.h"
#include "../Fluxes/CommonFluxMulti.h"
#include "../Allocators/AllocatorConstStep.h"

namespace Convolution
{
	/**
	 * @brief Class controls time grid
	 * for a ConstStep regime
	 */
	struct TimePolicyConstStep : public TimePolicy
	{
		TimePolicyConstStep(double ht_) noexcept :
			TimePolicy{ -ht_ , 0.0},
			ht{ ht_ }
		{}

		void set_interval() noexcept
		{
			its_currentTime += ht;
			its_previousTimeReal += ht;
		}

	protected:
		const double ht;
	};

	template<size_t WellFluxCount>
	struct ConstStepWell :
		public KernelConstStep,
		public FluxConstStep
	{
		using Kernel = KernelConstStep;
		using Flux = FluxConstStep;

		using WellFluxMulti =
			CommonFluxMulti
			<
				typename Flux, BaseWellFlux,
				WellFluxCount
			>;

		/**
		 * @param spatial_size nmbr of segments 
		 * within a well
		 */
		ConstStepWell(
			size_t spatial_size,
			size_t frame_temporal_size,
			size_t temporal_size) :
			KernelConstStep
		{
			spatial_size,
			frame_temporal_size
		},
			FluxConstStep
		{
			spatial_size,
			temporal_size,
			frame_temporal_size
		}
		{}
	};

	template<size_t WellFluxCount>
	struct ConstStepFrac :
		public ConstStepWell<WellFluxCount>
	{
		template<typename Allocator_t>
		using CommonFluxMulti_Alloc = 
			CommonFluxMulti<
				Allocator_t,
				BaseFracFlux,
				WellFluxCount
			>;

		using FracFluxMultiContainer = FracturesFluxContainer_t<
			typename Flux, CommonFluxMulti_Alloc>;

		std::vector<typename ConstStepWell::Kernel> fracKernelRegime;
		std::vector<typename ConstStepWell::Flux> fracFluxRegime;

		/**
		 * @param well_spatial_size nmbr of segments
		 * within a well
		 * 
		 * @param temporal_size total nmbr of time
		 * moments to be allocated for fluxes. 
		 * 
		 * @param frame_temporal_size total nmbr of 
		 * time moments to be allocated for 
		 * influence functions. This parameter takes 
		 * into account the external boundary.
		 * frame_temporal_size <= temporal_size
		 * 
		 * @param fracNy vector of nmbrs of y-nodes
		 * along the fractures
		 */
		ConstStepFrac(
			size_t well_spatial_size,
			size_t frame_temporal_size,
			size_t temporal_size,
			const std::vector<size_t>& fracNy) :

			ConstStepWell{ 
			well_spatial_size,
			frame_temporal_size, // for kernel
			temporal_size // for flux,
		}
#pragma region CTOR BODY
		{
			size_t frac_nmbr = fracNy.size();

			fracKernelRegime.reserve(frac_nmbr);
			fracFluxRegime.reserve(frac_nmbr);

			for (size_t frac_id = 0; frac_id < frac_nmbr; ++frac_id)
			{
				fracKernelRegime.emplace_back(
					fracNy[frac_id], frame_temporal_size);
				fracFluxRegime.emplace_back(
					fracNy[frac_id], temporal_size,
					frame_temporal_size);
			}
		}
#pragma endregion
	};

	template<size_t WellFluxCount>
	struct ConstStepPolicy :
		public ConstStepFrac<WellFluxCount>,
		public TimePolicyConstStep
	{
		using TimePolicy = TimePolicyConstStep;

		ConstStepPolicy(
			const ConstStepFrac<WellFluxCount>& constStep,
			const TimePolicyConstStep& timePolicy) noexcept :
			ConstStepFrac<WellFluxCount>{ constStep },
			TimePolicyConstStep{ timePolicy }
		{}
	};

	template<size_t WellFluxCount>
	using ConstStep =
		ConstStepPolicy<WellFluxCount>;
} // Convolution

