/*****************************************************************//**
 * \file   AllocatorSmallStep.h
 * \brief  The file contains allocator definitions for
 * the SmallStep calculations within the VarStep regime
 * 
 * It specifies data access (struct SmallStep)
 * on read (OnGet...) and write (OnPush...)
 * for kernels (struct KernelSmallStep)
 * and fluxes (struct FluxSmallStep)
 * 
 * \author artur.salamatin
 * \date   June 2023
 *********************************************************************/

#pragma once
#include "../Allocators/AllocatorSmallStep.h"

namespace Convolution
{
	struct TimePolicySmallStep : public TimePolicyConstStep
	{
		using TimePolicyConstStep::TimePolicyConstStep;
	};

	template<size_t WellFluxCount>
	struct SmallStepFrac : public ConstStepFrac<WellFluxCount>
	{
		using ConstStepFrac::ConstStepFrac;
	};

	template<size_t WellFluxCount>
	struct SmallStepPolicy :
		public SmallStepFrac<WellFluxCount>,
		public TimePolicySmallStep
	{
		using TimePolicy = TimePolicySmallStep;

		SmallStepPolicy(
			const SmallStepFrac& constStep,
			const TimePolicySmallStep& timePolicy) noexcept :
			SmallStepFrac{ constStep },
			TimePolicySmallStep{ timePolicy }
		{}
	};

	template<size_t WellFluxCount>
	using SmallStep = SmallStepPolicy<WellFluxCount>;
}

