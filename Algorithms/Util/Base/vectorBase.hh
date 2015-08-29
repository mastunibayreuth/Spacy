#ifndef ALGORITHM_VECTOR_BASE_HH
#define ALGORITHM_VECTOR_BASE_HH

#include "Algorithms/Util/Mixins/eps.hh"
#include "Algorithms/Util/Mixins/impl.hh"
#include "Algorithms/Util/Exceptions/incompatibleSpaceException.hh"
#include "Algorithms/vectorSpace.hh"

namespace Algorithm
{
  /// @cond
  template <class V>
  void checkSameSpaces(const V& x, const V& y);
  /// @endcond

  /**
   * @brief Base class for vector implementations.
   *
   * Provides access to the underlying vector space and related operations.
   * Stores an object of type Value. This object holds the employed vector implementation.
   * @see Real::Vector
   */
  template <class VType, class Value = void>
  class VectorBase :
      public VectorBase<VType> , public Mixin::Impl<Value>
  {
  public:
    VectorBase( const VectorSpace& space , const Value& value )
      : VectorBase<VType>(space) , Mixin::Impl<Value>(value)
    {}
  };


  /**
   * @brief Base class for vector implementations.
   * @see Kaskade::Vector, Fenics::Vector
   *
   * Provides access to the underlying vector space and related operations.
   */
  template <class VType>
  class VectorBase<VType,void>
  {
  public:
    /**
     * @brief Constructor.
     * @param space underlying vector space
     */
    VectorBase( const VectorSpace& space )
      : space_(space)
    {}

    /// Copy constructor.
    VectorBase(const VectorBase &other)
      : space_(other.space_)
    {}

    /// Move constructor.
    VectorBase(VectorBase&& other) noexcept
      : space_(other.space_)
    {}

    /// Copy assignment.
    VectorBase& operator=(const VectorBase& other)
    {
      checkSameSpaces(*this,other);
      return *this;
    }

    /// Move assignment.
    VectorBase& operator=(VectorBase&& other) noexcept
    {
      checkSameSpaces(*this,other);
      return *this;
    }

    /// Access underlying vector space.
    const VectorSpace* space() const
    {
      return &space_;
    }

    /// Access index of underlying space.
    unsigned spaceIndex() const
    {
      return space()->index();
    }

    /// Compute norm of vector. The norm defined in the underlying vector space is employed.
    double norm() const
    {
      return space()->norm()(static_cast<const VType&>(*this));
    }

    /// Compute scalar product \f$(x,y)\f$. The scalar product defined in the underlying vector space is employed.
    double scalarProduct(const VType& y) const
    {
      return space()->scalarProduct()( static_cast<const VType&>(*this) , y );
    }

  private:
    const VectorSpace& space_;
  };

  /**
   * @brief Base class providing some operations for vectors via CRTP.
   * @see Real::Vector, Fenics::Vector, Kaskade::Vector
   *
   * Provides operator+=, operator-=, operator*=, operator-, operator==.
   */
  template <class VType>
  class SupportedOperatorBase : public Mixin::Eps
  {
  public:
    /// Compute \f$ x+=y\f$.
    VType& operator+=(const VType& y)
    {
      checkSameSpaces(static_cast<const VType&>(*this),y);
      static_cast<VType*>(this)->impl() += y.impl();
      return static_cast<VType&>(*this);
    }

    /// Compute \f$ x-=y\f$.
    VType& operator-=(const VType& y)
    {
      checkSameSpaces(static_cast<const VType&>(*this),y);
      static_cast<VType*>(this)->impl() -= y.impl();
      return static_cast<VType&>(*this);
    }

    /// Compute \f$a*=x\f$.
    VType& operator*=(double a)
    {
      static_cast<VType*>(this)->impl() *= a;
      return static_cast<VType&>(*this);
    }

    /// Get \f$-x\f$.
    VType operator-() const
    {
      VType y = static_cast<const VType&>(*this);
      y.impl() *= -1;
      return y;
    }

    /// Comparison operators.
    bool operator==(const VType& y) const
    {
      checkSameSpaces(static_cast<const VType&>(*this),y);
      auto dx = y;
      dx -= static_cast<const VType&>(*this);
      return dx(dx) < eps()*eps();
    }
  };
}

#endif // ALGORITHM_VECTOR_BASE_HH