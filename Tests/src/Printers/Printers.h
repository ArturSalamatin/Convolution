#pragma once
#include <iostream>

#include "Convolvers/ConvolutionDefines.h"
#include "Convolvers/Allocators/AllocatorConstStep.h"
#include "Convolvers/Kernels/BaseKernel.h"

namespace Tests
{
	using BaseKernelConstStep =
		Convolution::BaseKernel
		<
		Convolution::KernelConstStep
		>;

	std::ostream& operator<<(
		std::ostream& o, 
		const BaseKernelConstStep& kernel
		);
	std::ostream& operator<<(
		std::ostream& o, const Convolution::OnGetKernelConstStep& kernel);
	std::ostream& operator<<(
		std::ostream& o, const Convolution::OnPushKernelConstStep& kernel);
	std::ostream& operator<<(
		std::ostream&, const Convolution::OnGetFluxConstStep&);
	std::ostream& operator<<(
		std::ostream&, const Convolution::KernelConstStep&);
	std::ostream& operator<<(
		std::ostream& o, const Convolution::MemoryDesc& memDesc);

	template<typename T>
	void print(std::ostream& o, const T& flux)
	{
		o
			<< static_cast<Convolution::MemoryDesc>(flux)
			<< "Start index in container:                       "
			<< flux.idx_begin() << '\n'
			<< "End index in container:                         "
			<< flux.idx_end() << '\n';
	}
}

class Printers
{
};

