#pragma once

// Algorithms
#include <Spacy/Algorithm/Newton/newton.hh>
#include <Spacy/Algorithm/Newton/terminationCriteria.hh>

// Spaces
#include <Spacy/Spaces/realSpace.hh>
#include <Spacy/Spaces/productSpace.hh>

// Util
#include <Spacy/Util/cast.hh>
#include <Spacy/Util/copy.hh>
#include <Spacy/Util/invoke.hh>
#include <Spacy/Util/log.hh>
#include <Spacy/Util/mixins.hh>
#include <Spacy/Util/voider.hh>

// Interfaces and directly related functionality
#include <Spacy/functional.hh>
#include <Spacy/c1Functional.hh>
#include <Spacy/c2Functional.hh>
#include <Spacy/derivative.hh>
#include <Spacy/operator.hh>
#include <Spacy/c1Operator.hh>
#include <Spacy/linearOperator.hh>
#include <Spacy/linearSolver.hh>
#include <Spacy/scalarProduct.hh>
#include <Spacy/inducedScalarProduct.hh>
#include <Spacy/norm.hh>
#include <Spacy/hilbertSpaceNorm.hh>
#include <Spacy/vector.hh>
#include <Spacy/vectorSpace.hh>
#include <Spacy/zeroVectorCreator.hh>
