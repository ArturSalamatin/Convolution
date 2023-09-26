#include "Printers.h"

namespace Tests
{
	template<>
	void print<Convolution::KernelConstStep>(
		std::ostream& o, const Convolution::KernelConstStep& kernel)
	{
		o
			<< kernel.pusher
			<< kernel.extractor;
	}
	template<>
	void print<BaseKernelConstStep>(
		std::ostream& o, const BaseKernelConstStep& kernel
		)
	{
		o
			<< "Number of rows in the Kernel-matrix:            "
			<< kernel.rows() << '\n'
			<< "Number of filled-in cols in the Kernel-matrix:  "
			<< kernel.cols() << '\n'

			<< kernel.allocator;
	}

	std::ostream& operator<<(
		std::ostream& o,
		const BaseKernelConstStep& kernel
		)
	{
		o
			<< "The BaseKernel<KernelConstStep> state is:\n";
		print(o, kernel);
		return o;
	}

	std::ostream& operator<<(
		std::ostream& o, const Convolution::KernelConstStep& flux)
	{
		o
			<< "The KernelConstStep state is:\n";
		print(o, flux);
		return o;
	}

	std::ostream& operator<<(
		std::ostream& o, const Convolution::OnGetFluxConstStep& flux)
	{
		o
			<< "The OnGetFluxConstStep state is:\n";
		print(o, flux);
		return o;
	}

	std::ostream& operator<<(
		std::ostream& o, const Convolution::OnPushKernelConstStep& flux)
	{
		o
			<< "The OnPushKernelConstStep state is:\n";
		print<Convolution::OnPushKernelConstStep>(o, flux);
		return o;
	}

	std::ostream& operator<<(
		std::ostream& o, const Convolution::OnGetKernelConstStep& flux)
	{
		o
			<< "The OnGetKernelConstStep state is:\n";
		print(o, flux);
		return o;
	}

	std::ostream& operator<<(
		std::ostream& o, const Convolution::MemoryDesc& memDesc)
	{
		o
			<< "The MemroyDesc state is:\n"
			<< "Number of sources in the problem:               "
			<< memDesc.spatial_size() << '\n'
			<< "Number of time steps to be simulated:           "
			<< memDesc.temporal_size() << '\n'
			<< "Number of space x time sources in the problem:  "
			<< memDesc.allocated_memory() << '\n';

		return o;
	}
}
