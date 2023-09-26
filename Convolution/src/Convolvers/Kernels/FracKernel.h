#pragma once
#include "BaseKernel.h"

namespace Convolution
{
	/**
	 * @brief Container for a SINGLE 
	 * fracture-related kernel, 
	 * which stores coefs R and U and
	 * calculates the sum(R*(U-U)).
	 */
	template<typename Allocator_t>
	class FracKernel :
		public BaseKernel<Allocator_t/*=KernelConstStep*/>
	{
	public:
		FracKernel(
			size_t nodesCount,
			const typename KernelTypedefs<Allocator_t>::Allocator& convDesc) :
			BaseKernel{
			nodesCount, convDesc }
		{}

		using BaseKernel<Allocator_t>::push_coef;
		void push_coef(
			const double* R_data, const double* U_data)
		{
			//		F.colwise() = ArrayXd::Map(R_data, block_height());
					// P_cur should be filled in in-place
					// now, it is copied, which is not optimal
			P_cur = ArrayXXd::Map(U_data, block_height(), block_width());
			// calculate a new block and ADD it to Kernel,
			// at appropriate positions
			Kernel.middleCols(
				block_stride_in_row(), block_width()) +=
				(
					(P_cur - P_prev).colwise() *
					ArrayXd::Map(R_data, block_height())
				).matrix();

			P_prev = std::move(P_cur);

#ifdef PUSHER_ADVANCE_FLAG
			allocator.pusher.need_advance = true;
#endif
		}

		void push_coef_prev(
			const double* U_data)
		{
			//		F.colwise() = ArrayXd::Map(R_data, block_height());
					// P_cur should be filled in in-place
					// now, it is copied, which is not optimal
			P_prev = ArrayXXd::Map(U_data, block_height(), block_width());

#ifdef PUSHER_ADVANCE_FLAG
			allocator.pusher.need_advance = true;
#endif
		}

		void advance()
		{
			// prepare the initial state for the next time moment
			on_advance();
		}
		void reset_kernel()
		{
			// prepare the initial state for the next time moment
			Kernel = MatrixXd::Zero(Kernel.rows(), Kernel.cols());
		}
	};

	/**
	 * @brief Container class for a set of 
	 * fracture-related kernels.
	 * 
	 * An object of such class corresponds to a 
	 * single group of nodes and multiple fractures.
	 */
	template<typename Allocator_t>
	class FracKernelContainer :
		public KernelTypedefs<Allocator_t>,
		public MultipleFracturesContainer<FracKernel<Allocator_t>>
	{
		// current time index
		size_t nt;
	public:
		FracKernelContainer() = default;

		FracKernelContainer(
			const std::vector<typename
			KernelTypedefs<Allocator_t>::Allocator>&
			vec_convDesc,
			size_t nodesCount) :
			MultipleFracturesContainer<
			FracKernel<Allocator_t>>{
			vec_convDesc.size()
		},
			nt{ 0 }
		{
			for (size_t frac = 0; frac < vec_convDesc.size(); ++frac)
			{
				data.emplace_back(
					nodesCount,
					vec_convDesc[frac]
				);
			}
		}

		void push_coef(
			const double* R_data, 
			const double* U_data)
		{
			data[cur_frac_id].push_coef(R_data, U_data); // push to the current fracture
			this->on_push_coef(); // advance to the next fracture in container in a closed loop
		}
		void push_coef_prev(
			const double* U_data)
		{
			data[cur_frac_id].push_coef_prev(U_data); // push to the current fracture
			this->on_push_coef(); // advance to the next fracture in container in a closed loop
		}
		void reset_kernel() noexcept
		{
			data[cur_frac_id].reset_kernel();
		}

		void advance()
		{
			for (auto& k : data)
				k.advance();
#ifdef PUSHER_ADVANCE_FLAG
			need_advance = false;
#endif
		}


		/**
		* Pushing coefficients to the
		* fracture kernel container is a
		* complex multistep procedure.
		* Even for a single fracture.
		*
		* Therefore, a special method
		* is introduced to indicate that
		* pushing to a single fracture is done.
		*
		* \param R_data Values of R-function
		* (with z corrected by zT)
		* \param U_data Values of U-function
		*/
		void push_done()
		{
			++nt;
			cur_frac_id = (1 + cur_frac_id) % frac_count; // advance to the next fracture in container in a closed loop
		}

		double Irs(
			size_t frac_id, size_t frac_node,
			size_t l, size_t nt) const
		{
			return data[frac_id](l, frac_node, nt);
		}
	};
} // Convolution
