#ifndef ALGORITHM_FUNCTION_SPACES_PRODUCTSPACE_HH
#define ALGORITHM_FUNCTION_SPACES_PRODUCTSPACE_HH

#include <memory>
#include <vector>

#include "../../Interface/abstractHilbertSpace.hh"

namespace Algorithm
{
  class AbstractNorm;
  class AbstractScalarProduct;
  class AbstractFunctionSpaceElement;

  /// Space of real numbers.
  class ProductSpace : public AbstractHilbertSpace
  {
  public:
    ProductSpace(std::vector<std::unique_ptr<AbstractBanachSpace> >&& primalSpaces);

    ProductSpace(std::vector<std::unique_ptr<AbstractBanachSpace> >&& primalSpaces,
                 std::vector<std::unique_ptr<AbstractBanachSpace> >&& dualSpaces);

    const std::vector<std::unique_ptr<AbstractBanachSpace> >& getPrimalSpaces() const;

    const std::vector<std::unique_ptr<AbstractBanachSpace> >& getDualSpaces() const;

  private:
    void setNormImpl(std::shared_ptr<AbstractNorm>) override;

    void setScalarProductImpl(std::shared_ptr<AbstractScalarProduct>) override;

    std::shared_ptr<AbstractScalarProduct> getScalarProductImpl() const override;

    std::unique_ptr<AbstractFunctionSpaceElement> elementImpl() const override;

    std::shared_ptr<AbstractScalarProduct> sp_;

    std::vector<std::unique_ptr<AbstractBanachSpace> > primalSpaces_;
    std::vector<std::unique_ptr<AbstractBanachSpace> > dualSpaces_;
  };

  template <class... Spaces_> struct PrimalSpaces {};
  template <class... Spaces_> struct DualSpaces {};

  namespace ProductSpaceDetail
  {
    template <class...> struct CreateSpaceVectorImpl;

    template <>
    struct CreateSpaceVectorImpl<>
    {
      static std::vector<std::unique_ptr<AbstractBanachSpace> > apply()
      {
        return std::vector<std::unique_ptr<AbstractBanachSpace> >();
      }
    };

    template <class Space, class... Spaces>
    struct CreateSpaceVectorImpl<Space,Spaces...>
    {
      static std::vector<std::unique_ptr<AbstractBanachSpace> > apply()
      {
        auto spaces = CreateSpaceVectorImpl<Spaces...>::apply();
        spaces.push_back(std::make_unique<Space>());
        return spaces;
      }
    };

    template <class> struct CreateSpaceVector;

    template <class... Spaces_>
    struct CreateSpaceVector< PrimalSpaces<Spaces_...> > : CreateSpaceVectorImpl<Spaces_...> {};

    template <class... Spaces_>
    struct CreateSpaceVector< DualSpaces<Spaces_...> > : CreateSpaceVectorImpl<Spaces_...> {};
  }


//  template <unsigned nPrimal, unsigned nDual, class... Spaces>
  template <class PrimalSpaces, class DualSpaces=DualSpaces<> >
  std::unique_ptr<ProductSpace> makeProductSpace()
  {
    return std::make_unique<ProductSpace>( ProductSpaceDetail::CreateSpaceVector<PrimalSpaces>::apply(),
                                           ProductSpaceDetail::CreateSpaceVector<DualSpaces>::apply() );
  }
}

#endif // ALGORITHM_FUNCTION_SPACES_PRODUCTSPACE_HH