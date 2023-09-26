#pragma once
#include "BaseFluxContainer.h"

namespace Convolution
{
	/**
	 * @brief Provides the logic of data addition to the
	 * FluxContainer.
	 *
	 * It is specified for wells, where the flux-log is divided
	 * by the permeability-log.
	 */
	template<typename Allocator_t>
	class BaseWellFlux :
		public BaseFluxContainer<Allocator_t>
	{
	public:
		using BaseFluxContainer<Allocator_t>::BaseFluxContainer;
		using BaseFluxContainer<Allocator_t>::push_coef;

		/**
		 * \brief Method pushes the qzi/permeability ratio at a new time moment
		 */
		void push_coef(const double* cur_qzi, const double* perm)
		{
			on_push();
			flux.segment(
				allocator.pusher.idx_begin(),
				allocator.pusher.spatial_size()) =
				calc_coef(cur_qzi, perm);
		}

		auto calc_coef(
			const double* cur_qzi,
			const double* perm)
		{
			return ArrayXd::Map(cur_qzi, allocator.pusher.spatial_size()) /
				ArrayXd::Map(perm, allocator.pusher.spatial_size());
		}
	};

} // Convolution
