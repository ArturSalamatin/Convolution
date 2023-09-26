#pragma once
#include <vector>
#include <cassert>

/**
* @brief This flag is for debug purpose only.
* It switches on the checks whether the P, E, F, R, U
* matricies have been rewrittn or not.
*/
#define _PUSHER_ADVANCE_FLAG

namespace Convolution
{
	/**
	 * @brief 
	 * Common abstract descriptor of allocated memory 
	 * for source data (well, fracture) and 
	 * influence functions data (ConvolutionKernel).
	 *
	 * @param spatial_size
	 * overall number of mass sources,
	 * i.e., well or fracture segments..
	 *
	 * @param temporal_size
	 * overall time frames that will be finally saved in container.
	 * It is either overall nmbr of time frames (for qzi fluxes),
	 * or nmbr required to get to the external boundary, 
	 * for 1-P : F-E : R-U coefs.
	 */
	struct MemoryDesc
	{
		MemoryDesc(const MemoryDesc&) noexcept = default;
		MemoryDesc(size_t spatial_size_,
			size_t temporal_size_) noexcept :
			its_spatial_size{ spatial_size_ },
			its_temporal_size{ temporal_size_ },
			its_allocated_memory{ spatial_size_ * temporal_size_ },
			cur_temporal_window{ 0 }
		{}

		size_t spatial_size() const noexcept
		{
			return its_spatial_size;
		}
		size_t temporal_size() const noexcept
		{
			return its_temporal_size;
		}
		size_t allocated_memory() const noexcept
		{
			return its_allocated_memory;
		}

	protected:
		const size_t its_allocated_memory;
		const size_t its_spatial_size;
		const size_t its_temporal_size;
		// size of the memory that is
		// considered filled to the next
		// push/extract operation
		size_t cur_temporal_window;
	};

	/**
	 * @brief Common abstract base class for
	 * getters(extractors). 
	 * This class hierarchy is responsible
	 * for maintaining the correct state of container
	 * on EXTRACT(GET) operation.
	 */
	struct GetDesc : public MemoryDesc
	{
		GetDesc(const MemoryDesc& memoryDesc) noexcept :
			MemoryDesc{ memoryDesc }
		{}

		// the FOUR methods declare
		// a common interface for all
		// derived concrete implementations
		// of get-classes

		// on_extract: the data is going to be pulled from
		// Container.
		// This fact must be fixed
		// by updating descriptors before
		// the data is pulled.
		// So, now we get a proper
		// set of coefficients 
		// in a later call
		void on_extract() noexcept = delete;
		size_t idx_begin() const noexcept = delete;
		size_t idx_end() const noexcept = delete;
		// this method implementation is added by the class
		// template <typename push, typename extract>
		// struct Allocator
		size_t current_window_size() const noexcept = delete;
	};

	/**
	 * @brief Common abstract base class for 
	 * pushers. This class hierarchy is responsible
	 * for maintaining the correct state of container
	 * on PUSH operation.
	 */
	struct PushDesc : public MemoryDesc
	{
		PushDesc(const MemoryDesc& memoryDesc) noexcept :
			MemoryDesc{ memoryDesc }
#ifdef PUSHER_ADVANCE_FLAG
			,
			need_advance{ false }
#endif
		{}

#ifdef PUSHER_ADVANCE_FLAG
		bool is_correct_state() const noexcept
		{
			return need_advance;
		}
#endif

		size_t pushed_data_counter() const noexcept
		{
			return MemoryDesc::cur_temporal_window;
		}
		size_t push_data_nmbr() const noexcept
		{
			return MemoryDesc::temporal_size();
		}

		// the three methods declare
		// a common interface for all
		// derived concrete implementations
		// of push-classes
		void on_push() noexcept = delete;
		size_t idx_begin() const noexcept = delete;
		size_t idx_end() const noexcept = delete;
#ifdef PUSHER_ADVANCE_FLAG
		bool need_advance;
#endif
	};

	/**
	 * @brief Common abstract allocator for Kernel and Flux
	 * and various regimes (ConstStep, MainStep, SmallStep)
	 * that manages
	 * the pushing (push-class) and extraction (extract-class)
	 * of data in and from the Container.
	 */
	template <typename push, typename extract>
	struct Allocator
	{
		using Push = push;
		using BaseExtract = extract;

		/**
		 * @brief Every Extractor must return the 
		 * total nmbr of coefficients taken for convolution.
		 * 
		 * This logic is imposed explicitly in the allocator.
		 */
		struct Extract : public BaseExtract
		{
			Extract(const BaseExtract& extractor) :
				BaseExtract{ extractor }
			{}

			size_t current_window_size() const noexcept {
				return BaseExtract::idx_end() - BaseExtract::idx_begin();
			}
		};

		Push pusher;
		Extract extractor;

		Allocator(const Push& pusher,
			const BaseExtract& extractor) :
			pusher{ pusher },
			extractor{ Extract{extractor} }
		{}
		size_t pushed_data_counter() const noexcept
		{
			return pusher.pushed_data_counter();
		}
		size_t push_data_nmbr() const noexcept
		{
			return pusher.push_data_nmbr();
		}
	};


	struct TimePolicy
	{
		TimePolicy(
			double its_previousTimeReal,
			double its_currentTime) noexcept :
			its_previousTimeReal{ its_previousTimeReal },
			its_currentTime{ its_currentTime }
		{}

		double currentTime() noexcept
		{
			return its_currentTime;
		}
		double previousTimeReal() noexcept
		{
			return its_previousTimeReal;
		}
		void set_interval() noexcept = delete;

	protected:
		double its_currentTime;
		double its_previousTimeReal;
	};

	/**
	 * @brief Container to store either all fluxes
	 * or all kernels related to fractures
	 */
	template<typename data_type>
	class MultipleFracturesContainer
	{
	protected:
		std::vector<data_type> data;
		const 
		size_t frac_count; // total number of fractures
		size_t cur_frac_id; // the fracture id to be pushed to,
		bool need_advance;
	public:
		MultipleFracturesContainer() = default;
		/**
		 * \brief Ctor that reserves memory for a given 
		 * number of fractures. The allocation is performed 
		 * in the anscestor classes.
		 * 
		 * \param frac_count The number of fractures that 
		 * must be stored
		 */
		MultipleFracturesContainer(
			size_t frac_count) :
			frac_count{ frac_count },
			cur_frac_id{ 0 },
			need_advance{ false }
		{
			data.reserve(frac_count);
		}

		/**
		 * \brief Get a fracture-relate element
		 * from the container.
		 * 
		 * \param frac_id Element (fracture) id
		 * \return Data related to a particular fracture
		 */
		const data_type& operator[] (size_t frac_id) const
		{
			assert(frac_id < frac_count);
			return data[frac_id];
		}
		/**
		 * \brief The number of elements in the container
		 */
		size_t size() const
		{
			return frac_count;
		}
		
	protected:
		/**
		 * \brief Once the data is pushed to a new fracture, 
		 * the state of the fracture is not self-consistent. 
		 */
		void on_push_coef()
		{
			need_advance = true;
		}
	};
} // Convolution
