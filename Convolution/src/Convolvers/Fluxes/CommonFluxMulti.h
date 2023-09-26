#pragma once
#include <array>
#include <Eigen/Dense>
#include <Eigen/Core>

namespace Convolution
{
	using namespace Eigen;

	/**
	* @brief The class performs convolution
	* with a set of multiple kernels simultaneously.
	* 
	* @tparam Flux_t Stands for the regime
	* of the WellFlux usage, i.e., 
	* BaseWellFluxMainStep,
	*		or 
	* BaseWellFlux 
	* 
	*		OR
	* 
	* BaseFracFluxMainStep,
	*		or 
	* BaseFracFlux 
	*/
	template<
		typename Allocator_t,
		template<typename Allocator_t> typename Flux_t, 
		size_t array_size>
	struct CommonFluxMulti : public Flux_t<Allocator_t>
	{
		template<typename kernel_type>
		using container_type = 
			std::array<kernel_type, array_size>;

	protected:
		std::array<VectorXd, array_size> 
			convolved_data_vector;

	public:
		// This is a general purpose ctor.
		CommonFluxMulti(
			const typename Flux_t<Allocator_t>::Allocator& convDesc) :
			Flux_t<Allocator_t>{ convDesc }
		{}

		template<typename kernel_type>
		const std::array<VectorXd, array_size>& convolve(
			const container_type<kernel_type>& kernels)
		{
			// data() method calls the 
			// on_extract method.
			// It must be called only once 
			// per convolution of multiple 
			// kernels with the same flux data
			/*const BaseWellFlux<
				typename Purpose::Allocator::Flux>& */
			decltype(auto) data = Flux_t<Allocator_t>::extract();
			for (size_t id = 0; id < array_size; ++id)
			{
				// using data() we take an appropriate flux set
				// for convolution of MainStep terms at various 
				// small steps.
				convolved_data_vector[id] = 
					data.convolve(kernels[id]);
			}
			return convolved_data_vector;
		}

		double result(size_t idx, size_t data_id)
		{
			return (*this)[data_id][idx];
		}

		constexpr size_t size() const
		{
			return array_size;
		}

		size_t size(size_t id) const
		{
			return convolved_data_vector[id].size();
		}

		const VectorXd& operator[](size_t data_id) const
		{
			return convolved_data_vector[data_id];
		}
	};
} // Convolution
