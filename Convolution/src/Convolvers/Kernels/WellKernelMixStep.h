#pragma once
#include <vector>
#include "WellKernel.h"
#include "../Allocators/AllocatorMixStep.h"

namespace Convolution
{
	/**
	* @brief The class is to store WellKernel coefficients for the MixStep regime.
	* Here, the E and P coefficients are calculated in SmallStep and MainStep regimes.
	* So, they just need to be cached.
	* And only F coefficients need to be calculated.
	*/
	template<>
	class WellKernel<KernelMixStep> :
		public AdvancedWellKernel<KernelMixStep>
	{
	private:
		std::vector<Eigen::ArrayXXd> 
			// container stores the P/E matricies 
			// corresponding to 
			// to the MainStep step size
			Pcur_cache
			// container stores the P/E matricies 
			// corresponding to 
			// to the SmallStep step size
		//	, Pprev_cache
			;
		// which coeffs should be taken next
		// for Pcur and Pprev
		size_t get_Pcur_idx
			//, get_Pprev_ind
			;

		const size_t small_step_nmbr_per_main_step;
		size_t small_step_counter_within_main_step;

	public:
		WellKernel(
			size_t nodesCount,
			const KernelMixStep& convDesc) :
			AdvancedWellKernel<KernelMixStep>{ nodesCount, convDesc },
			get_Pcur_idx{ 0ull },
			// the last SmallStep within the MainStep is excluded
			// nothing is calculated at this time interval
			small_step_nmbr_per_main_step{ convDesc.small_step_nmbr_per_main_step - 1 },
			small_step_counter_within_main_step{ 0ull }
		{
			Pcur_cache.reserve(convDesc.M);
		//	Pprev_cache.reserve(convDesc.Pprev_cache_size());
		}
		
		// push MainStep time step coefficients
		template<typename Matrix>
		void push_Pcur(const Matrix& matrix)
		{
			if (Pcur_cache.capacity() == Pcur_cache.size())
			{
				throw std::exception("WellKernel<KernelMixStep>::push_Pprev: too much data cached!");
			}
			Pcur_cache.push_back(matrix);
		}

		using BaseKernel::advance;
		void advance()
		{
			if (small_step_counter_within_main_step % small_step_nmbr_per_main_step == 0)
			{
				if (get_Pcur_idx == Pcur_cache.size())
				{
					throw std::exception("WellKernel<KernelMixStep>::advance: next Pcur_cache-item is not available!");
				}
		//		BaseKernel::P_cur = Pcur_cache[get_Pcur_idx];
				++get_Pcur_idx;
			}
			++small_step_counter_within_main_step;
			small_step_counter_within_main_step %= small_step_nmbr_per_main_step;

			BaseKernel::advance();
		}
			   
	private:
		static bool is_equal(const Eigen::ArrayXXd& lhs, const Eigen::ArrayXXd & rhs)
		{
			if (lhs.rows() != rhs.rows() || lhs.cols() != lhs.cols())
				return false;

			for(size_t i = 0; i < lhs.rows(); ++i)
				for (size_t j = 0; j < lhs.cols(); ++j)
				{
					if (std::abs(lhs(i, j) - rhs(i, j)) / std::abs(lhs(i, j) + rhs(i, j)) > 1E-10)
						return false;
				}

			return true;
		}
	};

	// A class template specialization for the WellKernel
	// related to MixStep regime
//	using WellKernelMixStep = WellKernel<KernelMixStep>;
} // Convolution
