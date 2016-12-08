// This file was automatically generated by friendly type erasure.
// Please do not modify.

#include "c1Functional.hh"

namespace Spacy
{
    C1Functional::C1Functional() noexcept : impl_( nullptr )
    {
    }

    C1Functional::C1Functional( const C1Functional& other )
        : functions_( other.functions_ ), type_id_( other.type_id_ ), impl_( other.impl_ )
    {
        if ( !type_erasure_table_detail::is_heap_allocated( other.impl_.get(), other.buffer_ ) )
            other.functions_.clone_into( other.impl_.get(), buffer_, impl_ );
    }

    C1Functional::C1Functional( C1Functional&& other ) noexcept : functions_( other.functions_ ),
                                                                  type_id_( other.type_id_ )
    {
        if ( type_erasure_table_detail::is_heap_allocated( other.impl_.get(), other.buffer_ ) )
            impl_ = std::move( other.impl_ );
        else
            other.functions_.clone_into( other.impl_.get(), buffer_, impl_ );
        other.impl_ = nullptr;
    }

    C1Functional& C1Functional::operator=( const C1Functional& other )
    {
        functions_ = other.functions_;
        type_id_ = other.type_id_;
        impl_ = other.impl_;
        if ( !type_erasure_table_detail::is_heap_allocated( other.impl_.get(), other.buffer_ ) )
            other.functions_.clone_into( other.impl_.get(), buffer_, impl_ );
        return *this;
    }

    C1Functional& C1Functional::operator=( C1Functional&& other ) noexcept
    {
        type_id_ = other.type_id_;
        functions_ = other.functions_;
        if ( type_erasure_table_detail::is_heap_allocated( other.impl_.get(), other.buffer_ ) )
            impl_ = std::move( other.impl_ );
        else
            other.functions_.clone_into( other.impl_.get(), buffer_, impl_ );
        other.impl_ = nullptr;
        return *this;
    }

    C1Functional::operator bool() const noexcept
    {
        return impl_ != nullptr;
    }

    Real C1Functional::operator()( const Vector& x ) const
    {
        assert( impl_ );
        return functions_.call_const_Vector_ref( read(), x );
    }

    Vector C1Functional::d1( const Vector& x ) const
    {
        assert( impl_ );
        return functions_.d1_const_Vector_ref( read(), x );
    }

    const VectorSpace& C1Functional::domain() const
    {
        assert( impl_ );
        return functions_.domain( read() );
    }

    void* C1Functional::read() const noexcept
    {
        assert( impl_ );
        return impl_.get();
    }

    void* C1Functional::write()
    {
        assert( impl_ );
        if ( !impl_.unique() && type_erasure_table_detail::is_heap_allocated( impl_.get(), buffer_ ) )
            functions_.clone( read(), impl_ );
        return read();
    }
}