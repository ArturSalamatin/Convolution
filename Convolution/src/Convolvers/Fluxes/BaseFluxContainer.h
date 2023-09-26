#pragma once

#include <Eigen/Dense>
#include <Eigen/Core>
#include <array>

#include "../ConvolutionDefines.h"
#include "../Kernels/BaseKernel.h"

#undef OMPH_CODE			// use parallel version with openMP

#ifndef PPL_CODE    // parallel code, using ppl
#ifndef OMPH_CODE   // parallel code, using openMP
#define SEQUEN_CODE // sequential code, MatrixXd*VectorXd is not done in parallel
#endif
#endif


#ifdef OMPH_CODE
#include <omp.h>
#endif
#ifdef PPL_CODE
#include <ppl.h>
#endif

namespace Convolution
{
	using namespace Eigen;

	template<typename Allocator_t>
	struct FluxTypedefs
	{
		using Allocator = Allocator_t;
	};

	template<typename Allocator_t>
	class CommonBase : public FluxTypedefs<Allocator_t>
	{
	protected:
		void on_push() noexcept
		{
			allocator.pusher.on_push();
		}

		void on_extract() const noexcept
		{
			allocator.extractor.on_extract();
		}
	public:
		mutable FluxTypedefs<Allocator_t>::Allocator 
			allocator;

		size_t flux_push_counter() const noexcept
		{
			return allocator.pushed_data_counter();
		}

		size_t flux_push_nmbr() const noexcept
		{
			return allocator.push_data_nmbr();
		}

		CommonBase(const Allocator& convDesc) :
			allocator{convDesc}
		{}
		/**
		* \brief Returns the number of rows in the vector
		* with the stored and RELEVANT data. 
		* It does not consider the forgotten data
		* due to external boundary.
		* It is less than the overall allocated memory.
		*/
		size_t rows() const
		{
			return allocator.extractor.current_window_size();
		}
		/**
		 * \brief Returns the number of cols in the vector, which is exactly 1.
		 */
		constexpr size_t cols() const
		{
			return 1ull;
		}

		/**
		 * @brief The operator gives access to the flux
		 * corresponding to a particular time moment.
		 * 
		 * It must be overloaded in derived classes.
		 */
		double operator()(size_t nt, size_t segm_id) const = delete;

		/**
		 * @brief The method returns the current 
		 * flux container. It also applies the on_extract() method
		 *
		 * It must be overloaded in derived classes.
		 */
		const auto& extract() const = delete;

		auto operator()() const = delete;
	};

	/**
	 * \brief Container to store the flux data
	 * which is to be convolved with the Kernel.
	 * The data is stored continuously in memory.
	 * 
	 * Only its ancestors can accept the new data,
	 * since the way it is pushed depends on the 
	 * nature of the flux and source.
	 */
	template<typename Allocator_t>
	class BaseFluxContainer : public CommonBase<Allocator_t>
	{
	protected:
		// a ColumnMajor vector of Nwell*Nt(rows) by 1(cols) elements
		VectorXd flux;

	public:
		BaseFluxContainer(
			const typename KernelTypedefs<Allocator_t>::Allocator& 
			convDesc) :
				CommonBase<Allocator_t>{ convDesc },
				flux{ VectorXd::Zero(convDesc.pusher.allocated_memory()) }
		{
#ifdef OMPH_CODE
			// check whether the threads have already been created 
			// for openMP
			if (omp_get_num_threads() < omp_get_num_procs())
			{
				// create threads if have not been created un advance
				omp_set_dynamic(0);
				omp_set_num_threads(omp_get_num_procs());
			}
#endif
		}

		/**
		 * \brief Returns the flux-data for a linear source term
		 * which is associated with a segment and a time frame.
		 *
		 * The data is not exactly the flux. The flux can be modified on push,
		 * i.e., divided by permeability of the matrix.
		 *
		 * \param nt time frame for the flux data
		 * \param segm_id segment of the linear source term associated with the flux-data
		 */
		double operator()(size_t nt, size_t segm_id) const
		{
			return flux(segm_id + flux.size() - nt * allocator.extractor.spatial_size());
		}
		/**
		 * \brief Returns the entire flux data 
		 * already pushed to the container.
		 * It is an Eigen::VectorBlock of size (rows; cols)
		 */
		auto operator()() const
		{
			return flux.segment(
				allocator.extractor.idx_begin(), 
				rows());
		}

		/**
		 * \brief Method to convolve the BaseKernel
		 * with the BaseFluxContainer column for all the mesh points at once
		 *
		 * \return Result of convolution for all mesh points
		 */
		template<typename KernelAllocator_t>
		VectorXd convolve(
			const BaseKernel<KernelAllocator_t>& kernel) const
		{
#ifdef OMPH_CODE
			////////////////////////////////////////////////////openMP version
			// nmbr of rows in the matrix/Kernel
			size_t rows = kernel.rows();
			// memory allocation for the result of convolution
			VectorXd out{ rows };
			// total nmbr of available threads
			// created in advance, once and for all
			size_t real_thread_count = omp_get_num_threads(); // omp_get_num_procs();
			// set the number of threads
	//		size_t temp = omp_get_num_threads();
			// nmbr of rows to be convolved by a single thread
			size_t g = rows / real_thread_count + 1;
			// number of thread that will  be used in the code
			ptrdiff_t used_thread_count = rows / g + 1;

#pragma omp parallel for //num_threads(used_thread_count)
			for (ptrdiff_t idx = 0; idx < used_thread_count; ++idx)
			{
				// nmbr of rows to be convolved in a single thread
				size_t count = (std::min)(rows, (idx + 1) * g) - idx * g;
				// std::string str{ std::to_string(omp_get_num_threads()) };
				out.segment(idx * g, count) =
					kernel().middleRows(idx * g, count) * (*this)();
			}

			return out;
#else
#ifdef SEQUEN_CODE
			return //VectorXd::Ones(kernel.rows());
				kernel()
				//		MatrixXd::Ones(kernel.rows(), kernel.cols()) 
				* (*this)();
#else
#ifdef PPL_CODE
			std::static_assert("IMPLEMENT CONVOLUTION USING PPL LIBRARY");
#else
			std::static_assert("IMPLEMENT CONVOLUTION");
#endif
#endif
#endif
		}

		const BaseFluxContainer<Allocator_t>& extract() const
		{
			CommonBase<Allocator_t>::
				on_extract();
			return *this;
		}


		template<typename T>
		void push_coef(const T& data)
		{
			on_push();
			flux.segment(
				allocator.pusher.idx_begin(),
				allocator.pusher.spatial_size()) = data;
		}
	};
} // Convolution

