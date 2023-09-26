#pragma once

#include "../Allocators/AllocatorMixStep.h"

namespace Convolution
{
	template<size_t WellFluxCount>
	struct MixStepWell :
		public KernelMixStep,
		public FluxMixStep
	{
		using Kernel = KernelMixStep;
		using Flux = FluxMixStep;

		using WellFluxMulti =
			CommonFluxMulti<
			typename Flux,
			BaseWellFlux,
			Grids::AllNodeGroups::nodes_id::size
			>;

		MixStepWell(
			size_t spatial_size,
			size_t frame_temporal_size,
			size_t small_step_nmbr_per_main_step,
			size_t M) :
			Kernel
		{
			spatial_size,
			frame_temporal_size,
			small_step_nmbr_per_main_step,
			M
		},
			Flux
		{
			spatial_size,
			frame_temporal_size
		}
		{}
	};

	template<size_t WellFluxCount>
	struct MixStepFrac :
		public MixStepWell<WellFluxCount>
	{
		template<typename Allocator_t>
		using CommonFluxMulti_Alloc =
			CommonFluxMulti<Allocator_t,
			BaseFracFlux,
			Grids::AllNodeGroups::nodes_id::size>;

		using FracFluxMultiContainer =
			FracturesFluxContainer_t<
			typename Flux, CommonFluxMulti_Alloc>;

		std::vector<typename MixStepWell::Kernel>
			fracKernelRegime;
		std::vector<typename MixStepWell::Flux>
			fracFluxRegime;

		static constexpr size_t frame_temporal_size{1};

		/**
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
		MixStepFrac(
			size_t well_spatial_size,
			size_t small_step_nmbr_per_main_step,
			// number of main steps within the second part of history
			size_t M,
			const std::vector<size_t>& fracNy
		) :
			MixStepWell
			{
				well_spatial_size,
				this->frame_temporal_size, // only a single term participates in the convolution
				small_step_nmbr_per_main_step,
				M
			}
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
					this->frame_temporal_size,
					small_step_nmbr_per_main_step,
					M);
				fracFluxRegime.emplace_back(
					fracNy[frac_id],
					this->frame_temporal_size);
			}
		}
	};

	struct TimePolicyMixStep :
		public TimePolicy
	{
		TimePolicyMixStep(
			size_t small_step_nmbr_per_main_step_,
			double main_step_) noexcept :
			TimePolicy{ 0.0, 0.0 },
			small_step_nmbr_per_main_step{ small_step_nmbr_per_main_step_ },
			main_step{ main_step_ },
			small_step{ main_step_ / small_step_nmbr_per_main_step_ },
			small_step_counter_within_main_step{ 0ull }
		{}

		void set_interval() noexcept
		{
			if (
				small_step_counter_within_main_step % small_step_nmbr_per_main_step == 0ull)
			{
				TimePolicy::its_currentTime += main_step;
			}
			// the last segment is entered
			// skip it and move to the next mainStep
			TimePolicy::its_previousTimeReal += small_step;

			++small_step_counter_within_main_step;
			small_step_counter_within_main_step %= small_step_nmbr_per_main_step;
		}

	protected:
		const size_t small_step_nmbr_per_main_step;
		const double main_step;
		const double small_step;
		size_t small_step_counter_within_main_step;
	};

	template<size_t WellFluxCount>
	struct MixStepPolicy :
		public MixStepFrac<WellFluxCount>,
		public TimePolicyMixStep
	{
		using TimePolicy = TimePolicyMixStep;

		MixStepPolicy(const MixStepFrac& mixStep,
			const TimePolicy& timePolicy) noexcept :
			MixStepFrac{ mixStep },
			TimePolicy{ timePolicy }
		{}
	};

	template<size_t WellFluxCount>
	using MixStep = MixStepPolicy<WellFluxCount>;

} // Convolution
