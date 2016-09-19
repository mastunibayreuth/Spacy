// This file was automatically generated by friendly type erasure.
// Please do not modify.

#include "linearOperator.hh"

namespace Spacy
{
    LinearOperator::LinearOperator() noexcept : impl_( nullptr )
    {
    }

    LinearOperator::LinearOperator( const LinearOperator& other ) : functions_( other.functions_ ), impl_( other.impl_ )
    {
    }

    LinearOperator::LinearOperator( LinearOperator&& other ) noexcept : functions_( other.functions_ ),
                                                                        type_id_( other.type_id_ )
    {
        if ( type_erasure_table_detail::is_heap_allocated( other.impl_.get(), other.buffer_ ) )
            impl_ = std::move( other.impl_ );
        else
            other.functions_.clone_into( other.impl_.get(), buffer_, impl_ );
        other.impl_ = nullptr;
    }

    LinearOperator& LinearOperator::operator=( const LinearOperator& other )
    {
        functions_ = other.functions_;
        type_id_ = other.type_id_;
        impl_ = other.impl_;
        return *this;
    }

    LinearOperator& LinearOperator::operator=( LinearOperator&& other ) noexcept
    {
        functions_ = other.functions_;
        if ( type_erasure_table_detail::is_heap_allocated( other.impl_.get(), other.buffer_ ) )
            impl_ = std::move( other.impl_ );
        else
            other.functions_.clone_into( other.impl_.get(), buffer_, impl_ );
        other.impl_ = nullptr;
        return *this;
    }

    LinearOperator::operator bool() const noexcept
    {
        return impl_ != nullptr;
    }

    Vector LinearOperator::operator()( const Vector& x ) const
    {
        assert( impl_ );
        return functions_.call_const_Vector__ref_( *this, read(), x );
    }

    Real LinearOperator::operator()( const LinearOperator& x ) const
    {
        assert( impl_ );
        return functions_.call_const_LinearOperator__ref_( *this, read(), x.impl_.get() );
    }

    LinearOperator& LinearOperator::operator+=( const LinearOperator& y )
    {
        assert( impl_ );
        return functions_.add_const_LinearOperator__ref_( *this, write(), y.impl_.get() );
    }

    LinearOperator& LinearOperator::operator-=( const LinearOperator& y )
    {
        assert( impl_ );
        return functions_.subtract_const_LinearOperator__ref_( *this, write(), y.impl_.get() );
    }

    LinearOperator& LinearOperator::operator*=( double a )
    {
        assert( impl_ );
        return functions_.multiply_double_( *this, write(), std::move( a ) );
    }

    LinearOperator LinearOperator::operator-() const
    {
        assert( impl_ );
        return functions_.negate( *this, read() );
    }

    bool LinearOperator::operator==( const LinearOperator& y ) const
    {
        assert( impl_ );
        return functions_.compare_const_LinearOperator__ref_( *this, read(), y.impl_.get() );
    }

    std::function< Vector( const Vector& ) > LinearOperator::solver() const
    {
        assert( impl_ );
        return functions_.solver( *this, read() );
    }

    const VectorSpace& LinearOperator::domain() const
    {
        assert( impl_ );
        return functions_.domain( *this, read() );
    }

    const VectorSpace& LinearOperator::range() const
    {
        assert( impl_ );
        return functions_.range( *this, read() );
    }

    const VectorSpace& LinearOperator::space() const
    {
        assert( impl_ );
        return functions_.space( *this, read() );
    }

    void* LinearOperator::read() const noexcept
    {
        assert( impl_ );
        return impl_.get();
    }

    void* LinearOperator::write()
    {
        if ( !impl_.unique() )
        {
            if ( type_erasure_table_detail::is_heap_allocated( impl_.get(), buffer_ ) )
                functions_.clone( impl_.get(), impl_ );
            else
                functions_.clone_into( impl_.get(), buffer_, impl_ );
        }
        return impl_.get();
    }

    LinearSolver operator^( const LinearOperator& A, int k )
    {
        if ( k == -1 )
            return A.solver();
        throw Exception::InvalidArgument( "operator^ for LinearOperator only defined for exponent: k = -1." );
    }

    LinearSolver operator^( LinearOperator&& A, int k )
    {
        if ( k == -1 )
            return std::move( A.solver() );
        throw Exception::InvalidArgument( "operator^ for LinearOperator only defined for exponent: k = -1." );
    }

    LinearOperator& axpy( LinearOperator& A, double a, LinearOperator B )
    {
        return A += ( B *= a );
    }
}
