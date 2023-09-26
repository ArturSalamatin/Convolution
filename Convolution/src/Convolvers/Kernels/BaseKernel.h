#pragma once
#include <cassert>
#include <string>
#include <exception>

#include <Eigen/Core>
#include <Eigen/Dense>
#include "../ConvolutionDefines.h"

namespace Convolution
{
	using namespace Eigen;

	/**
	 * @brief The class is required to save the 
	 * template type and use it in derived classes.
	 */
	template<typename Allocator_t /* = Kernel{Const, Main, Small, Mix}Step */>
	struct KernelTypedefs
	{
		using Allocator = Allocator_t; /* = Kernel{Const, Main, Small, Mix}Step */
	};

	/**
	 * @brief Class provides basic interface 
	 * to allocated memory and allows for
	 * access of the coefficients.
	 */
	template<typename Allocator_t>
	class BaseKernel : public KernelTypedefs<Allocator_t>
	{
	protected:
		void on_advance() noexcept
		{
			allocator.pusher.on_push();
		}

		void on_extract() const noexcept
		{
			allocator.extractor.on_extract();
		}

	public:
		/**
		 * \brief A ColMajor matrix which is convolved with fluxes
		 * Its columns are filled in with the products
		 * F*(P_cur - P_prev)
		 */
		MatrixXd Kernel; 
		/**
		 * \brief P-coefficients at a previous time step, 
		 *	size: number of mesh points (Ns*Ny*Nz) BY number of sources (well_nodes || frac_nodes)
		 */
		ArrayXXd P_prev;
		/**
		 * \brief P-coefficients at a previous time step,
		 * size: number of mesh points (Ns*Ny*Nz) BY number of sources (well_nodes || frac_nodes)
		 */
		ArrayXXd P_cur; 
		/**
		 * \brief F-coefficients for F(E-E)-product, 
		 * or F==1 if F(P-P)
		 */
		ArrayXXd F; 
		/**
		 * \brief Number of spatial grid nodes,
		 * number of rows in the Kernel
		 */
		const size_t grid_nodes_count;

		/**
		 * \brief Descriptor for convolution,
		 * it cotains the number of time frames to be stored in memory
		 * and the number of spatial source nodes, related to the kernel
		 */
		mutable Allocator_t allocator;

		// At every time-step a block of Kernel coefficients is pushed into
		// the block matrix to fill in the number of entire column.
		// The number of such columns per a time-step is convDesc.spatial_size
		// The block descriptors are as follows:
		
		// nmbr of rows in a block
		size_t block_height() const
		{
			return grid_nodes_count;
		}
		// nmbr of cols in a block
		size_t block_width() const
		{
			return allocator.pusher.spatial_size();
		}
		// some cols are filled in, 
		// so new columns to be filled 
		// are started here
		size_t block_stride_in_row() const
		{
			return allocator.pusher.idx_end();
		}
		// we always fill in the entire column
		constexpr size_t block_stride_in_col() const
		{
			return 0ull;
		}

		/**
		 * \brief Allocate memory for P_cur field
		 */
		void allocate_P_cur()
		{
			P_cur =
				ArrayXXd::Zero(
					block_height(),
					block_width());
		}
		/**
		 * \brief Verifies whether the state of the object is correct
		 * 
		 * Once the P,F (R,U) values are pushed into the object,
		 * the state becomes INcorrect.
		 */
		void is_correct_state() const
		{
#ifdef PUSHER_ADVANCE_FLAG
			if (allocator.pusher.is_correct_state())
				throw std::exception(
					"BaseKernel::is_correct_state() : The Kernel data cannot be accessed before its state is fixed with advance() method.");
#endif
		}
		/**
		 * \brief Returns the coefficient from the Kernel matrix
		 * at position (row; col).
		 * The method verifies the object state before it returns the value.
		 * 
		 * \param row Coefficient row
		 * \param col Coefficient col
		 * \return coefficient in a matrix
		 */
		double operator()(size_t row, size_t col) const
		{
			is_correct_state();
			return Kernel(row, col);
		}
	public:
		BaseKernel(
			size_t nodesCount,
			const typename KernelTypedefs<Allocator_t>::Allocator& convDesc) :
			Kernel{ nodesCount, convDesc.pusher.allocated_memory() },
			grid_nodes_count{ nodesCount },
			allocator{ convDesc }
		{
			// better to keep these initializations in the ctor body
			// since it calls class fields that may not be initialized
			// at a proper time moment
			P_prev = ArrayXXd::Zero(block_height(), block_width());
			F = ArrayXXd::Ones(block_height(), block_width());

			allocate_P_cur();
		}

		/**
		 * \brief The number of rows in the Kernel filled with data
		 */
		size_t rows() const
		{
			return block_height();
		}
		/**
		 * \brief The number of cols in the Kernel
		 * that are filled in with data
		 */
		size_t cols() const
		{
			return block_stride_in_row();
		}

		void push_coef(
			size_t row, size_t col, 
			double E, double f)
		{
			P_cur(row, col) = E;
			F(row, col) = f;
			allocator.pusher.need_advance = true;
		}
		/**
		 * \brief Returns the Matrix coefficient according to its physical meaning.
		 * 
		 * \param mesh_node_id Essentially a row in the matrix
		 * \param source_node_id Index of the finite element (segment) in a linear source
		 * \param time_node Index of the time moment
		 */
		double operator()(
			size_t mesh_node_id, 
			size_t source_node_id, 
			size_t time_node) const
		{
			is_correct_state();
			return (*this)(mesh_node_id, source_node_id + block_width() * time_node);
		}

		double get_P_cur(
			size_t mesh_node_id,
			size_t source_node_id) const
		{
			return P_cur(mesh_node_id, source_node_id);
		}

		const double* get_P_cur_ptr(
			size_t mesh_node_id,
			size_t source_node_id) const
		{
			return P_cur.data() + mesh_node_id + source_node_id * block_height();
		}

		const double* get_P_prev_ptr(
			size_t mesh_node_id,
			size_t source_node_id) const
		{
			return P_prev.data() + mesh_node_id + source_node_id * block_height();
		}

		const auto get_P_prev_block(
			size_t mesh_node_id,
			size_t source_node_id,
			size_t rows_count) const
		{
			return P_prev.block(
				mesh_node_id, source_node_id,
				rows_count,
				block_width());
		}

		double get_P_prev(
			size_t node_id,
			size_t source_node_id) const
		{
			return P_prev(node_id, source_node_id);
		}
		double get_F(
			size_t node_id,
			size_t source_node_id) const
		{
			return F(node_id, source_node_id);
		}

		const auto& get_P_prev() const
		{
			return P_prev;
		}
		const auto& get_P_cur() const
		{
			return P_cur;
		}

		/**
		 * \brief Returns the block in memory 
		 * which is filled in with the appropiate data
		 * 
		 * \return Matrix block
		 */
		auto operator()() const
		{
			is_correct_state();
			// this is for ConstStep
			// when large steps are not split into smaller steps.
			// Here, begin index of the coef-frame does not move.
			// return Kernel.leftCols(cols());

			// this is a more general approach
			// when initial large steps can be split into 
			// smaller steps
			// and the coef-frame moves,
			// its end idx as well as begin index
			on_extract();
			return Kernel.middleCols(
				allocator.extractor.idx_begin(),
				allocator.extractor.current_window_size());
		}

		/**
		 * \brief Method is responsible for advance in time at a single time step.
		 *
		 * It is assumed that P_prev already exists,
		 * P_cur has been filled in. Thus, P_cur - P_prev is
		 * calculated here. The difference fills in 
		 * the respective columns in Kernel.
		 */
		void advance()
		{
			// calculate a new block and send it to Kernel,
			// at appropriate positions
			Kernel.middleCols(
				block_stride_in_row(), block_width()) =
				F * (P_cur - P_prev);

			P_prev = std::move(P_cur);
			allocate_P_cur();
			// prepare the initial state for the next time moment
			// fix the current state
			on_advance();
		}
	};

	template<typename Allocator_t>
	class BaseKernelFile : public BaseKernel<Allocator_t>
	{
	public:
		BaseKernelFile(
			size_t nodesCount,
			const Allocator_t& convDesc,
			const std::string& kernelName) 
			: 
			BaseKernel{nodesCount, convDesc}
		{}

		void advance()
		{
			// you can print here on advance
			BaseKernel<Allocator_t>::advance();
		}
	};
} // Convolution
