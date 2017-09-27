// Copyright (C) 2015 by Lars Lubkoll. All rights reserved.
// Released under the terms of the GNU General Public License version 3 or later.

#pragma once

#include "linalg/direct.hh"
#include <Spacy/zeroVectorCreator.hh>
#include "Spacy/vector.hh"
#include "Spacy/Util/cast.hh"
#include "Spacy/Util/Base/OperatorBase.hh"
#include "util.hh"
#include "vectorSpace.hh"
#include "linalg/symmetricOperators.hh"



namespace Spacy
{
namespace Kaskade
{

/**
 * @ingroup KaskadeGroup
 * @brief Wrapper class to call apply-routines from different Kaskade preconditioners the same way (is to be unified in Kaskade)
 * @param Prec  %Kaskade Preconditioner (i.e., %Kaskade::AdditiveMultiGrid < > )
 * @param range range space of the preconditioner
 * @param domain domain space of the preconditioner
 */

template<class Range, class Domain, class Prec>
class Apply
{

public:
	// const Refs ?
	static void wrap(Prec & prec, Range & y, Domain & x)
	{
		prec.apply(y, x);
	}

};
/**
 * @ingroup KaskadeGroup
 * @brief Wrapper class to call apply-routines from different Kaskade preconditioners the same way (is to be unified in Kaskade)
 * @param Prec  %Kaskade Preconditioner (i.e., %Kaskade::AdditiveMultiGrid < > )
 * @param range range space of the preconditioner
 * @param domain domain space of the preconditioner
 */

// TODO
// Domain Range Vertauscht ???
template<class Range, class Domain, class T, class U, class V>
class Apply<Range, Domain, ::Kaskade::AdditiveMultiGrid<T, U, V> >
{
public:
	static void wrap(::Kaskade::AdditiveMultiGrid<T, U, V> & prec, Range & y,
			Domain & x)
	{
		prec.apply(::Kaskade::component<0>(y), ::Kaskade::component<0>(x));
	}

};





/** NOT TESTED
 * @ingroup KaskadeGroup
 * @brief Simple Jacobi Preconditioner
 * @param range range space of the preconditioner
 * @param domain domain space of the preconditioner
 */
template<class Domain, class Range, class KaskadeOperator>
class SimpleJacobiPreconditioner: public SymmetricPreconditioner<Domain, Range>
{
public:

	using Base = SymmetricPreconditioner<Domain,Range>;

	SimpleJacobiPreconditioner() = default;

	SimpleJacobiPreconditioner(const SimpleJacobiPreconditioner &) = default;

	// Not nice
	SimpleJacobiPreconditioner(SimpleJacobiPreconditioner &&) = default;

	SimpleJacobiPreconditioner(const KaskadeOperator &A) :
			init_(true)
	{

		const auto n = A.get().N();
		diag_.resize(n, 0.0);
		const auto size = A.get().size();
		const auto & rows = A.get().ridx;
		const auto & cols = A.get().cidx;
		const auto & data = A.get().data;

		size_t i = 0;

		while (i < size)
		{
			// use iterators
			if (rows[i] == cols[i])
			{
				diag_[rows[i]] = data[i];
			}

			i++;
		}

	}

	 virtual void apply( Domain & domain, const Range & range) override
	{

//


	auto & x = domain;

	const auto & b = range;

//		auto & x = component<0>(domain);
//
//		const auto & b = component<0>(range);
//
		size_t size = x.size();
//
////		std::cout << "TestJacobiApplyVectorLength: " <<  b.N() << " " << x.N() << '\n';
////		std::cout << "TestJacobiApplyVectorLength: " <<  b.size() << " " << x.size() << '\n';
////		std::cout << "TestJacobiApplyVectorLength: " <<  b.dim() << " " << x.dim() << '\n';
//		// comment
//
		assert(init_ && size == b.size() && x.dim() == b.dim() == diag_.size());


//		std::vector<double> tmpx, tmpb;
//		::Kaskade::domainToVector(domain, domain);
//		::Kaskade::rangeToVector(range, tmpb);
//		size_t size = domain.size();
//
//		for(size_t i = 0; i < size; i++)
//			tmpx[i] = tmpb[i]/diag_[i];


		auto xIter = x.begin();
		auto diagIter = diag_.cbegin();

		for (const auto & it : b)
		{
			auto vecIter = xIter->begin();

			for (const auto & entry : it)
			{
				*(vecIter) = entry / (*diagIter);
				diagIter++;
				vecIter++;
			}
			xIter++;
		}


	}

	  typename Base::field_type applyDp(Domain& x, Range const& r) override
	        {

	         apply(x,r);

//	         auto & xLoc = component<0>(x);
//	         const auto & rLoc = component<0>(r);

	         return x*r;
	        }

	  bool requiresInitializedInput() const
	         {



	           return false;
	         }




	void init(const KaskadeOperator &A)
	{
;
		init_ = true;
		const auto n = A.get().N();
		diag_.resize(n, 0.0);
		const auto size = A.get().size();
		const auto & rows = A.get().ridx;
		const auto & cols = A.get().cidx;
		const auto & data = A.get().data;

		size_t i = 0;

		while (i < size)
		{
			// use iterators
			if (rows[i] == cols[i])
			{
				diag_[rows[i]] = data[i];
			}
			i++;
		}

	}

private:
// not well implemented

	std::vector<double> diag_;
	bool init_ = false;

};




template<class Range, class Domain, class KaskadeOperator>
class Apply<Range, Domain, ::Spacy::Kaskade::SimpleJacobiPreconditioner< Range, Domain, KaskadeOperator > >
{
public:
	static void wrap(::Spacy::Kaskade::SimpleJacobiPreconditioner< Range, Domain, KaskadeOperator > & prec, Range & y,
			Domain & x)
	{
		prec.apply(::Kaskade::component<0>(y), ::Kaskade::component<0>(x));
	}

};



/**
 * @ingroup KaskadeGroup
 * @brief Preconditioner interface for %Kaskade 7.
 */
template<class AnsatzVariableDescription, class TestVariableDescription,
		class Prec>
class Preconditioner: public OperatorBase
{
	using Spaces = typename AnsatzVariableDescription::Spaces;
	using Domain = typename AnsatzVariableDescription::template CoefficientVectorRepresentation<>::type;
	using Range = typename TestVariableDescription::template CoefficientVectorRepresentation<>::type;
public:
	Preconditioner() = delete;
	/**
	 * @brief Constructor.
	 * @param spaces boost fusion forward sequence of space pointers required to initialize temporary storage
	 * @param domain domain space of the solver
	 * @param range range space of the solver
	 */
	Preconditioner(const VectorSpace& domain, const VectorSpace& range,
			Prec& prec) :
			OperatorBase(domain, range), prec_(prec), spaces_(
					extractSpaces<AnsatzVariableDescription>(domain))
	{
	}

	/// Compute \f$A^{-1}x\f$.
	::Spacy::Vector operator()(const ::Spacy::Vector& x) const
	{

		// use range and domain defined by the preconditioner
		Range y_(
				TestVariableDescription::template CoefficientVectorRepresentation<>::init(
						spaces_));
		Domain x_(
				AnsatzVariableDescription::template CoefficientVectorRepresentation<>::init(
						spaces_));
		copyToCoefficientVector<AnsatzVariableDescription>(x, x_);

		// Changed Range and Domain in Kaskade
		Apply<Range, Domain, Prec>::wrap(prec_, y_, x_);

		auto y = ::Spacy::zero(range());
		copyFromCoefficientVector<TestVariableDescription>(y_, y);

		return y;
	}

private:
	Prec& prec_;
	Spaces spaces_;

};

/**
 * @brief Convenient generation of a Spacy preconditioner from a Kaskade preconditioner
 * @param A %Kaskade operator (i.e., AssembledGalerkinOperator or MatrixRepresentedOperator)
 * @param spaces boost fusion forward sequence of space pointers required to initialize temporary storage
 * @param domain domain space of the solver
 * @param range range space of the solver
 * @param Prec Kaskade preconditioner (e.g. Kaskade::AdditiveMultiGrid < > )
 * @return Preconditioner<AnsatzVariableSetDescription, TestVariableSetDescription, Preconditioner  >( domain , range , preconditioner )
 */
template<class AnsatzVariableSetDescription, class TestVariableSetDescription,
		class Prec>
auto makePreconditioner(const VectorSpace& domain, const VectorSpace& range,
		Prec& prec)
{
	return Preconditioner<AnsatzVariableSetDescription,
			TestVariableSetDescription, Prec>(domain, range, prec);
}

}
}
