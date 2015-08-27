#ifndef ALGORITHM_FUNCTIONS_1D_QUADRATIC_FUNCTION_HH
#define ALGORITHM_FUNCTIONS_1D_QUADRATIC_FUNCTION_HH

namespace Algorithm
{
  namespace Functions_1D
  {
    /**
     * @ingroup AlgorithmGroup
     * @brief A one-dimensional quadratic function \f$q(t) = a + bt + ct^2\f$.
     */
    class Quadratic
    {
    public:
      /**
       * @brief Constructor.
       * @param a coefficient of constant term
       * @param b coefficient of linear term
       * @param c coefficient of quadratic term
       */
      Quadratic(double a, double b, double c) noexcept;

      /// Compute \f$q(t) = a + bt + ct^2 \f$.
      double operator()(double t) const noexcept;

    private:
      double a_, b_, c_;
    };
  }
}

#endif // ALGORITHM_FUNCTIONS_1D_QUADRATIC_FUNCTION_HH
