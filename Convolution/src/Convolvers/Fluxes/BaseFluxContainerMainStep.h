#pragma once
#include <Eigen/Dense>
#include <Eigen/Core>
#include <array>

#include "WellFlux.h"
#include "FracFlux.h"

namespace Convolution
{
	using namespace Eigen;

	template<
		typename Allocator_t,  
		template<typename Allocator_t> typename Flux_t>
	struct BaseFluxContainerMainStep :
		public FluxTypedefs<Allocator_t>
	{
		/**
		 * @brief ctor
		 */
		BaseFluxContainerMainStep(
			const Allocator_t& convDesc) :
			small_step_nmbr{ convDesc.small_step_nmbr },
			main_step_counter{ 0ull },
			main_step_nmbr{ convDesc.main_step_nmbr },
			// a flux of all zeros is required initially 
			// for the averaging
			prev_flux{ ArrayXd::Zero(convDesc.pusher.allocated_memory()) }, 
			flux_set
		{
			std::vector<Flux_t<Allocator_t>>(
				convDesc.small_step_nmbr,
				Flux_t<Allocator_t>{convDesc})
		}
		{
			// This flux should be used for convolution.
			// Initially it is a non-averaged flux.
			// At the second period of history,
			// the pointer shall switch to other 
			// elements of flux_set-vector
			// in a circular fashion
			cur_container_id = convDesc.small_step_nmbr - 1;
			flux_ptr = &flux_set.back();
		}


		size_t flux_push_counter() const noexcept
		{
			return flux_ptr->allocator.pushed_data_counter();
		}

		size_t flux_push_nmbr() const noexcept
		{
			return flux_ptr->allocator.push_data_nmbr();
		}

		/**
		 * @brief Advance the fluxContainer to the next one
		 */
		void switch_fluxContainer()
		{
			// modulo division
			cur_container_id = (cur_container_id + 1) % small_step_nmbr;
			flux_ptr = &flux_set[cur_container_id];
		}

		/**
		 * @brief Set a particular fluxContainer
		 */
		void switch_fluxContainer(size_t step_id)
		{
			assert(step_id < small_step_nmbr);
			cur_container_id = step_id;
			flux_ptr = &flux_set[cur_container_id];
		}

		/**
		 * @brief The method pushes data to
		 * flux container.
		 * 
		 * It takes into account that fractures and 
		 * well take different set of parameters,
		 * so Args... expands differently for fracs and well.
		 */
		template<typename... Args>
		void push_coef(Args... args)
		{
			auto qzi_to_perm = flux_ptr->calc_coef(args...).eval();

			// the back() element contains the "raw", unaveraged data
			flux_set.back().push_coef(qzi_to_perm);
			double local_small_step_counter = 1.0;
			for (auto& it = flux_set.begin(); it < flux_set.end() - 1; ++it)
			{
				double ratio =
					local_small_step_counter /
					static_cast<double>(small_step_nmbr);
				// push averaged fluxes to other flux_set-elements
				(*it).push_coef(
					ratio * qzi_to_perm +
					(1.0 - ratio) * prev_flux);
				local_small_step_counter += 1.0;
			}

			prev_flux = std::move(qzi_to_perm);
		}

		/**
		 * \brief The Logic of getting flux data
		 * for MainStep regime of calculation is not trivial.
		 *
		 * While we are in the first part of the history,
		 * we only push flux-data with averaging,
		 * and the exact, unaveraged data back.
		 *
		 * And in the second part of the history,
		 * we DO NOT push, and only take averaged data.
		 */
		const Flux_t<Allocator_t>& extract()
		{
			if (main_step_counter < main_step_nmbr)
			{
				++main_step_counter;
				for (auto& it : flux_set)
					it.extract();
			}
			else
			{
				switch_fluxContainer();
			}
			return (*flux_ptr);
		}

		double operator()(size_t nt, size_t segm_id) const
		{
			if (nt - 1 < main_step_nmbr)
				// first part of history
				return flux_set.back()(nt, segm_id);
			else
			{
				// second part of history
				return flux_set[(nt - 1 - main_step_nmbr) % small_step_nmbr]
				()(segm_id);
			}
		}

	protected:
		std::vector<Flux_t<Allocator_t>> flux_set;
		Flux_t<Allocator_t>* flux_ptr;

	protected:
		size_t main_step_counter;
		const size_t small_step_nmbr;
		size_t cur_container_id;

		const size_t main_step_nmbr;

		ArrayXd prev_flux;
	};

	template<typename Allocator_t>
	using BaseWellFluxMainStep = 
		BaseFluxContainerMainStep<Allocator_t, BaseWellFlux>;

	template<typename Allocator_t>
	using BaseFracFluxMainStep =
		BaseFluxContainerMainStep<Allocator_t, BaseFracFlux>;
} // Convolution
