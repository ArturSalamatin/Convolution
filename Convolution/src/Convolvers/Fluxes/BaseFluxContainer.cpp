#include "BaseFluxContainer.h"
#include "../Kernels/BaseKernel.h"
#include "../Allocators/AllocatorConstStep.h"
#include "../Allocators/AllocatorMainStep.h"




namespace Convolution
{
	using namespace Eigen;

#pragma region BaseFluxContainer
//	template<typename Allocator_t>
//	BaseFluxContainer<Allocator_t>::BaseFluxContainer(
//		const typename KernelTypedefs<Allocator_t>::Allocator& convDesc) :
//		CommonBase<Allocator_t>{ convDesc },
//		flux{ VectorXd::Zero(convDesc.pusher.allocated_memory()) }
//	{
//#ifdef OMPH_CODE
//		// check whether the threads have already been created 
//		// for openMP
//		if (omp_get_num_threads() < omp_get_num_procs())
//		{
//			// create threads if have not been created un advance
//			omp_set_dynamic(0);
//			omp_set_num_threads(omp_get_num_procs());
//		}
//#endif
//	}

	/*template<typename Allocator_t>
	double BaseFluxContainer<Allocator_t>::operator()(
		size_t nt, size_t s_node) const
	{
		return flux(s_node + flux.size() - nt * allocator.extractor.spatial_size());
	}*/

	//template<typename Allocator_t>
	//VectorXd BaseFluxContainer<Allocator_t>::convolve(
	//	const BaseKernel<CompatibleKernel>& kernel) const
	
#pragma endregion

//#pragma region BaseWellFlux
//	template<typename Allocator_t>
//	void BaseWellFlux<Allocator_t>::push_coef(
//		const double* cur_qzi, 
//		const double* perm)
//	{
//		on_push();
//		flux.segment(
//			allocator.pusher.idx_begin(),
//			allocator.pusher.spatial_size()) = 
//				calc_coef(cur_qzi, perm);
//	}
//#pragma endregion

#pragma region BaseFracFlux
	// fractures: 
	// method pushes qzf*/(per*hf) data at a new time moment
	//template<typename Allocator_t>
	//void BaseFracFlux<Allocator_t>::push_coef(
	//	const double* cur_qzf, 
	//	double value /* = per*hf*/)
	//{
	//	on_push();
	//	flux.segment(
	//		allocator.pusher.idx_begin(),
	//		allocator.pusher.spatial_size()) =
	//			calc_coef(cur_qzf, value);
	//}
#pragma endregion

#pragma region FracturesFluxContainer
	//template<typename Allocator_t>
	//void FracturesFluxContainer<Allocator_t>::push_coef(
	//	const double* cur_qzf, double value /* = perm*hf*/)
	//{
	//	data[cur_frac_id].push_coef(cur_qzf, value); // push to the current fracture
	//	need_advance = true; // this->on_push_coef(); // advance to the next fracture in container in a closed loop
	//	// pushing flux data is a simple single step process
	//	// so, an indicator that pushing to a single fracture is done
	//	// is triggered on every push
	//	cur_frac_id = (1 + cur_frac_id) % frac_count; // advance to the next fracture in container in a closed loop
	//}

	/*template<typename Allocator_t>
	void FracturesFluxContainer<Allocator_t>::is_correct_state()
	{
		if (cur_frac_id != 0)
			throw std::exception("The data was not pushed into every fracture. Cannot convolve safely.");
	}*/

	/*template<typename Allocator_t>
	double FracturesFluxContainer<Allocator_t>::result(size_t idx) const
	{
		if (size() > 0)
			return convolved_data(idx);
		else
			return 0.0;
	}*/
#pragma endregion

#pragma region Template Instantiations
	/*template
		class BaseFluxContainer<FluxConstStep>;
	template
		class BaseFluxContainer<FluxMainStep>;
	template
		class BaseWellFlux<FluxConstStep>;
	template
		class BaseWellFlux<FluxMainStep>;*/
	//template
	//	class FracturesFluxContainer<FluxConstStep>;
	//template
	//	class FracturesFluxContainer<FluxMainStep>;
#pragma endregion
} // Convolution