// This file was automatically generated by friendly type erasure.
// Please do not modify.

#include "dynamicOperator.hh"

namespace Spacy
{
    DynamicOperator::DynamicOperator() noexcept : impl_( nullptr )
    {
    }

    DynamicOperator::DynamicOperator( const DynamicOperator& other )
        : functions_( other.functions_ ), impl_( other.impl_ )
    {
    }

    DynamicOperator::DynamicOperator( DynamicOperator&& other ) noexcept : functions_( other.functions_ ),
                                                                           type_id_( other.type_id_ )
    {
        if ( type_erasure_table_detail::is_heap_allocated( other.impl_.get(), other.buffer_ ) )
            impl_ = std::move( other.impl_ );
        else
            other.functions_.clone_into( other.impl_.get(), buffer_, impl_ );
        other.impl_ = nullptr;
    }

    DynamicOperator& DynamicOperator::operator=( const DynamicOperator& other )
    {
        functions_ = other.functions_;
        type_id_ = other.type_id_;
        impl_ = other.impl_;
        return *this;
    }

    DynamicOperator& DynamicOperator::operator=( DynamicOperator&& other ) noexcept
    {
        functions_ = other.functions_;
        if ( type_erasure_table_detail::is_heap_allocated( other.impl_.get(), other.buffer_ ) )
            impl_ = std::move( other.impl_ );
        else
            other.functions_.clone_into( other.impl_.get(), buffer_, impl_ );
        other.impl_ = nullptr;
        return *this;
    }

    DynamicOperator::operator bool() const noexcept
    {
        return impl_ != nullptr;
    }

    Vector DynamicOperator::operator()( const Vector& x ) const
    {
        assert( impl_ );
        return functions_.call_const_Vector__ref_( *this, read(), x );
    }

    LinearOperator DynamicOperator::M() const
    {
        assert( impl_ );
        return functions_.M( *this, read() );
    }

    const VectorSpace& DynamicOperator::domain() const
    {
        assert( impl_ );
        return functions_.domain( *this, read() );
    }

    const VectorSpace& DynamicOperator::range() const
    {
        assert( impl_ );
        return functions_.range( *this, read() );
    }

    void* DynamicOperator::read() const noexcept
    {
        assert( impl_ );
        return impl_.get();
    }

    void* DynamicOperator::write()
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

    DynamicLinearOperator::DynamicLinearOperator() noexcept : impl_( nullptr )
    {
    }

    DynamicLinearOperator::DynamicLinearOperator( const DynamicLinearOperator& other )
        : functions_( other.functions_ ), impl_( other.impl_ )
    {
    }

    DynamicLinearOperator::DynamicLinearOperator( DynamicLinearOperator&& other ) noexcept
        : functions_( other.functions_ ),
          type_id_( other.type_id_ )
    {
        if ( type_erasure_table_detail::is_heap_allocated( other.impl_.get(), other.buffer_ ) )
            impl_ = std::move( other.impl_ );
        else
            other.functions_.clone_into( other.impl_.get(), buffer_, impl_ );
        other.impl_ = nullptr;
    }

    DynamicLinearOperator& DynamicLinearOperator::operator=( const DynamicLinearOperator& other )
    {
        functions_ = other.functions_;
        type_id_ = other.type_id_;
        impl_ = other.impl_;
        return *this;
    }

    DynamicLinearOperator& DynamicLinearOperator::operator=( DynamicLinearOperator&& other ) noexcept
    {
        functions_ = other.functions_;
        if ( type_erasure_table_detail::is_heap_allocated( other.impl_.get(), other.buffer_ ) )
            impl_ = std::move( other.impl_ );
        else
            other.functions_.clone_into( other.impl_.get(), buffer_, impl_ );
        other.impl_ = nullptr;
        return *this;
    }

    DynamicLinearOperator::operator bool() const noexcept
    {
        return impl_ != nullptr;
    }

    Vector DynamicLinearOperator::operator()( double t, const Vector& x ) const
    {
        assert( impl_ );
        return functions_.call_double__const_Vector__ref_( *this, read(), std::move( t ), x );
    }

    DynamicLinearOperator& DynamicLinearOperator::operator+=( const DynamicLinearOperator& y )
    {
        assert( impl_ );
        return functions_.add_const_DynamicLinearOperator__ref_( *this, write(), y.impl_.get() );
    }

    DynamicLinearOperator& DynamicLinearOperator::operator-=( const DynamicLinearOperator& y )
    {
        assert( impl_ );
        return functions_.subtract_const_DynamicLinearOperator__ref_( *this, write(), y.impl_.get() );
    }

    DynamicLinearOperator& DynamicLinearOperator::operator*=( double a )
    {
        assert( impl_ );
        return functions_.multiply_double_( *this, write(), std::move( a ) );
    }

    DynamicLinearOperator DynamicLinearOperator::operator-() const
    {
        assert( impl_ );
        return functions_.negate( *this, read() );
    }

    bool DynamicLinearOperator::operator==( const DynamicLinearOperator& y ) const
    {
        assert( impl_ );
        return functions_.compare_const_DynamicLinearOperator__ref_( *this, read(), y.impl_.get() );
    }

    std::function< Vector( const Vector& ) > DynamicLinearOperator::solver() const
    {
        assert( impl_ );
        return functions_.solver( *this, read() );
    }

    const VectorSpace& DynamicLinearOperator::domain() const
    {
        assert( impl_ );
        return functions_.domain( *this, read() );
    }

    const VectorSpace& DynamicLinearOperator::range() const
    {
        assert( impl_ );
        return functions_.range( *this, read() );
    }

    const VectorSpace& DynamicLinearOperator::space() const
    {
        assert( impl_ );
        return functions_.space( *this, read() );
    }

    void* DynamicLinearOperator::read() const noexcept
    {
        assert( impl_ );
        return impl_.get();
    }

    void* DynamicLinearOperator::write()
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

    DynamicC1Operator::DynamicC1Operator() noexcept : impl_( nullptr )
    {
    }

    DynamicC1Operator::DynamicC1Operator( const DynamicC1Operator& other )
        : functions_( other.functions_ ), impl_( other.impl_ )
    {
    }

    DynamicC1Operator::DynamicC1Operator( DynamicC1Operator&& other ) noexcept : functions_( other.functions_ ),
                                                                                 type_id_( other.type_id_ )
    {
        if ( type_erasure_table_detail::is_heap_allocated( other.impl_.get(), other.buffer_ ) )
            impl_ = std::move( other.impl_ );
        else
            other.functions_.clone_into( other.impl_.get(), buffer_, impl_ );
        other.impl_ = nullptr;
    }

    DynamicC1Operator& DynamicC1Operator::operator=( const DynamicC1Operator& other )
    {
        functions_ = other.functions_;
        type_id_ = other.type_id_;
        impl_ = other.impl_;
        return *this;
    }

    DynamicC1Operator& DynamicC1Operator::operator=( DynamicC1Operator&& other ) noexcept
    {
        functions_ = other.functions_;
        if ( type_erasure_table_detail::is_heap_allocated( other.impl_.get(), other.buffer_ ) )
            impl_ = std::move( other.impl_ );
        else
            other.functions_.clone_into( other.impl_.get(), buffer_, impl_ );
        other.impl_ = nullptr;
        return *this;
    }

    DynamicC1Operator::operator bool() const noexcept
    {
        return impl_ != nullptr;
    }

    Vector DynamicC1Operator::operator()( double t, const Vector& x ) const
    {
        assert( impl_ );
        return functions_.call_double__const_Vector__ref_( *this, read(), std::move( t ), x );
    }

    Vector DynamicC1Operator::d1( double t, const Vector& x, const Vector& dx ) const
    {
        assert( impl_ );
        return functions_.d1( *this, read(), std::move( t ), x, dx );
    }

    LinearOperator DynamicC1Operator::linearization( double t, const Vector& x ) const
    {
        assert( impl_ );
        return functions_.linearization( *this, read(), std::move( t ), x );
    }

    LinearOperator DynamicC1Operator::M() const
    {
        assert( impl_ );
        return functions_.M( *this, read() );
    }

    const VectorSpace& DynamicC1Operator::domain() const
    {
        assert( impl_ );
        return functions_.domain( *this, read() );
    }

    const VectorSpace& DynamicC1Operator::range() const
    {
        assert( impl_ );
        return functions_.range( *this, read() );
    }

    void* DynamicC1Operator::read() const noexcept
    {
        assert( impl_ );
        return impl_.get();
    }

    void* DynamicC1Operator::write()
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
}
