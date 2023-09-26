#pragma once
#include "BaseKernel.h"

namespace Convolution
{
	/**
	 * @brief The class provides an interface to push
	 * well-related data to kernel.
	 * 
	 * @tparam Allocator_t Type of the class that manages access 
	 * to the coefs container on push and on extract
	 */
	template<typename Allocator_t>
	class AdvancedWellKernel :
		public BaseKernelFile<Allocator_t>
	{
	public:
		AdvancedWellKernel(
			size_t nodesCount,
			const typename KernelTypedefs<Allocator_t>::Allocator& convDesc) :
			BaseKernelFile{ nodesCount, convDesc, "WellKernelAdvanced" }
		{}

		/**
		 * @brief The method pushes only F.
		 * 
		 * The data in F is either 1.0 for Reflections
		 * or
		 * some calculaed value for Poisson regime
		 *
		 * @param start position where the coefficients block need to be pasted
		 * @param count nmbr of coefs to be inserted
		 * @param f pointer to F-coefs to be inserted
		 */
		[[deprecated]]
		void push_F(
			size_t start, size_t count,
			const double* f)
		{
			F.middleRows(start, count) = ArrayXXd::Map(
				f,
				count,
				allocator.pusher.spatial_size());
#ifdef PUSHER_ADVANCE_FLAG
			allocator.pusher.need_advance = true;
#endif
		}

		void push_F_source(
			size_t col,
			const double* f)
		{
			F.col(col) = ArrayXd::Map(
				f, this->block_height());
#ifdef PUSHER_ADVANCE_FLAG
			allocator.pusher.need_advance = true;
#endif
		}

		/**
		 * @brief The method pushes E and F
		 * when a Poisson regime is on in
		 * the well.
		 *
		 * @param start position where the coefficients block need to be pasted
		 * @param count nmbr of coefs to be inserted
		 * @param E pointer to E-coefs to be inserted
		 * @param f pointer to F-coefs to be inserted
		 */
		[[deprecated]]
		void push_coef(
			size_t start, size_t count,
			const double* f, const double* E)
		{
			P_cur.middleRows(start, count) = 
				ArrayXXd::Map(
				E,
				count,
				allocator.pusher.spatial_size());
			push_F(start, count, f);
		}

		void push_source(
			size_t col, 
			const double* f, const double* E)
		{
			P_cur.col(col) = 
				ArrayXd::Map(
				E,
				this->block_height());
			push_F_source(col, f);
		}

		/**
		 * @brief One has to push to previous values of P, at P_prev,
		 * when a switch from Poisson and Reflection in the
		 * well convolution algorithm happened.
		 *
		 * @param start position where the coefficients block need to be pasted
		 * @param count nmbr of coefs to be inserted
		 * @param E pointer to E-coefs to be inserted
		 */
		[[deprecated]]
		void push_coef_prev(
			size_t start, size_t count,
			const double* E)
		{
			P_prev.middleRows(start, count) =
				ArrayXXd::Map(
					E,
					count,
					allocator.pusher.spatial_size());
#ifdef PUSHER_ADVANCE_FLAG
			allocator.pusher.need_advance = true;
#endif
		}

		void push_source_prev(
			size_t col,
			const double* E)
		{
			P_prev.col(col) =
				ArrayXd::Map(
					E,
					this->block_height());
#ifdef PUSHER_ADVANCE_FLAG
			allocator.pusher.need_advance = true;
#endif
		}

		/**
		 * @brief One has to push to previous values of P, at P_prev,
		 * when a switch from Poisson and Reflection in the
		 * well convolution algorithm happened.
		 * 
		 * This method pushes to a particular column of the P_prev container.
		 *
		 * @param start position where the coefficients block need to be pasted
		 * @param count nmbr of coefs to be inserted
		 * @param E pointer to E-coefs to be inserted
		 */
		[[deprecated]]
		void push_coef_prev(
			size_t start, size_t count, size_t col,
			const double* E)
		{
			P_prev.col(col).segment(start, count) =
				ArrayXd::Map(
					E, count);

#ifdef PUSHER_ADVANCE_FLAG
			allocator.pusher.need_advance = true;
#endif
		}
	};

	template<typename Allocator_t>
	class WellKernel : public AdvancedWellKernel<Allocator_t>
	{
	public:
		using AdvancedWellKernel::AdvancedWellKernel;
	};
} // Convolution

