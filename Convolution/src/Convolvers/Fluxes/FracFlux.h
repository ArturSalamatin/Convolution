#pragma once
#include "BaseFluxContainer.h"

namespace Convolution
{
	/**
	* @brief Container of fluxes stored continuously in memory.
	* The container is specified for fracture qzf data.
	*/
	template<typename Allocator_t>
	class BaseFracFlux :
		public BaseFluxContainer<Allocator_t>
	{
	public:
		using BaseFluxContainer<Allocator_t>::BaseFluxContainer;
		using BaseFluxContainer<Allocator_t>::push_coef;

		/**
		 * \brief Method pushes the qzf/(permeability*hf) ratio at a new time moment
		 */
		void push_coef(
			const double* cur_qzf, double value /* = per*hf*/)
		{
			on_push();
			flux.segment(
				allocator.pusher.idx_begin(),
				allocator.pusher.spatial_size()) =
				calc_coef(cur_qzf, value);
		}

		auto calc_coef(
			const double* cur_qzf,
			double value /* = per*hf*/)
		{
			return ArrayXd::Map(cur_qzf, allocator.pusher.spatial_size()) /
				value;
		}
	};


	/**
	 * @brief Container with Flux_t<Allocator_t> as elements.
	 * Each container element stores flux-data related
	 * to a particular fracture.
	 *
	 * @tparam Allocator_t
	 * @tparam Flux_t<Allocator_t> describes the logic of 
	 * flux push/extract per a fracture at various regimes
	 */
	template<
		typename Allocator_t,
		template<typename Allocator_t> typename Flux_t /* = BaseFracFlux, */>
	class FracturesFluxContainer_t :
		public MultipleFracturesContainer<Flux_t<Allocator_t>>,
		public FluxTypedefs<Allocator_t>
	{
		/**
		 * \brief Result of convolution of all BaseFracFlux with the
		 * corresponding FractrureKernel
		 */
		VectorXd convolved_data;

		/**
		 * \brief Verifies whether the state of Fractures Container is correct.
		 * The state is correct if the new data was pushed into every fracture.
		 * Thus the cur_frac_id must iterate through every fracture
		 * and return to the ZERO value.
		 */
		void is_correct_state()
		{
			if (cur_frac_id != 0)
				throw std::exception("The data was not pushed into every fracture. Cannot convolve safely.");
		}

	public:
		FracturesFluxContainer_t(
			const std::vector<typename
			KernelTypedefs<Allocator_t>::Allocator>&
			vec_convDesc) :
			MultipleFracturesContainer<Flux_t<Allocator_t>>{
				vec_convDesc.size()
		}
		{
			for (size_t frac = 0; frac < vec_convDesc.size(); ++frac)
				data.emplace_back(vec_convDesc[frac]);
		}

		/**
		 * \brief Pushes qzf-data to a new fracture
		 * and increases the fracture id to push to the next
		 * fracture on a new method call
		 *
		 * \param cur_qzf Pointer to the qzf-data
		 * \param value cur_qzf is divided by this value on push
		 */
		void push_coef(const double* cur_qzf, double value /* = perm*hf*/)
		{
			data[cur_frac_id].push_coef(cur_qzf, value); // push to the current fracture
			need_advance = true; // this->on_push_coef(); // advance to the next fracture in container in a closed loop
			// pushing flux data is a simple single step process
			// so, an indicator that pushing to a single fracture is done
			// is triggered on every push
			cur_frac_id = (1 + cur_frac_id) % frac_count; // advance to the next fracture in container in a closed loop
		}

		/**
		 * \brief Method convolves fracture fluxes with fracture kernels
		 *
		 * \param kernels Kernels corresponding to fractures. To be convolved with fracture fluxes
		 * \return Result of convolution, the sum between all fractures
		 */
		template<typename KernetType /*KernelAllocator_t*/>
		const auto& convolve(
			const KernetType
			//FracKernelContainer<KernelAllocator_t>
			& kernels)
		{
			is_correct_state();

			/*consider declaring
			auto*/ /* instead of VectorXd out*/
			convolved_data = data[0].extract().convolve(kernels[0]);
			for (size_t frac_id = 1; frac_id < frac_count; ++frac_id)
			{
				convolved_data += data[frac_id].extract().convolve(kernels[frac_id]);
			}
			/*without .eval() in the loop*/
//				/*out*/ convolved_data += (kernels[frac_id].data() * data[frac_id].data()).eval();
			/*then an expression to be evaluated will be constructed
			promoting further optimization*/
			/*convolved_data = out.eval();*/

			return convolved_data;
		}

		/**
		 * \brief Result of convolution for a particular spatial node
		 *
		 * \param idx Linear index of a spatial mesh node
		 */
		double result(size_t idx) const
		{
			if (size() > 0)
				return convolved_data(idx);
			else
				return 0.0;
		}

		double flux(size_t nt, size_t frac_id, size_t y_face) const
		{
			return (*this)[frac_id](nt, y_face);
		}
	};
} // Convolution
