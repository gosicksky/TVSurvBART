// Bit-exact replica of the TOP LEVEL of
//     boost::math::quadrature::gauss_kronrod<double, 61>::integrate(f, a, b)
// (integrate_non_adaptive_m1_1 plus the first subdivision test of
// recursive_adaptive_integrate) for several integrands that share their node
// evaluations, e.g. exp(A(t)) and g_k(t) * exp(A(t)) for every gradient
// component g_k.  The sampler's integrands are smooth on every exposure
// interval, so boost practically never subdivides; recording boost's 61 node
// times once (recordNodes) lets a caller cache everything that does not change
// between likelihood evaluations at those nodes and pay one exp() per node for
// all components together (fused).
//
// Exactness contract.  For each component, fused() either reproduces boost's
// result bit for bit (ok = true) or reports that boost would subdivide
// (ok = false), in which case the caller MUST fall back to boost for that
// component.  Every accumulation below is plain IEEE-754 arithmetic written in
// exactly boost's expression form and evaluation order,
//     kronrod_result += (fp + fm) * w;   gauss_result += (fp + fm) * wg;
// and the package is built with -ffp-contract=off (src/Makevars), so neither
// boost nor this replica is contracted into fused multiply-adds and the two
// agree on any IEEE-754 toolchain (clang or gcc, with or without FMA hardware).
// A build whose own CXXFLAGS override that flag (e.g. -ffp-contract=fast) may
// contract boost and the replica differently, so every user of this header
// verifies the replica against boost at run time before relying on it
// (survivalBART::gkSelfCheck, cpwbart_grid's startup checks) and falls back to
// boost on any mismatch.
#ifndef GUARD_gk61_h
#define GUARD_gk61_h

#include <boost/version.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>

// Boost releases whose gauss_kronrod<double, N> top level (integrate ->
// recursive_adaptive_integrate -> integrate_non_adaptive_m1_1) was checked
// against this replica: 1.86 (the release in use); 1.87 ships a byte-identical
// gauss_kronrod.hpp; in 1.75 and 1.84 (R's BH package) that code differs from
// 1.86 only by static_cast<Real> of the weight/abscissa tables, which are
// no-ops for Real = double (the replica reads the tables of the boost it is
// compiled against, so their values need not agree across releases).  Other
// releases keep the plain boost path.
#if BOOST_VERSION == 107500 || BOOST_VERSION == 108400 || BOOST_VERSION == 108600 || BOOST_VERSION == 108700
#define GK61_REPLICA_SUPPORTED 1
#else
#define GK61_REPLICA_SUPPORTED 0
#endif

namespace gk61 {

const int kNodes = 61;     // abscissae of the 61-point Kronrod rule (top level)
const int kMaxComp = 17;   // most integrands one fused() pass accumulates

typedef boost::math::quadrature::gauss_kronrod<double, kNodes> GK;

// Record, in boost's evaluation order, the kNodes node times boost's top level
// visits on (a, b), finite a < b.  The probe integrand is smooth and positive,
// so boost stops at the top level unless the interval is degenerate (boost then
// bisects and the function returns false: the caller keeps the boost path).
inline bool recordNodes(double a, double b, double* T) {
  int cnt = 0;
  auto rec = [&](double t) { if(cnt < kNodes) T[cnt] = t; cnt++; return std::exp(0.001 * t); };
  GK::integrate(rec, a, b);
  return cnt == kNodes;
}

// Fused top level of GK61 on an interval with half-width scale = (b - a) / 2
// (computed exactly as boost does) for nc <= kMaxComp integrands.
// vals(q, o) must write the nc integrand values at the q-th recorded node
// (q = 0 .. kNodes-1, boost's evaluation order) into o[0 .. nc-1]; it is called
// once per node, in that order.  On return out[c] is boost's top-level estimate
// of component c, and ok[c] tells whether boost accepts it without subdividing
// (then out[c] equals boost's result bit for bit).
template <class V>
inline void fused(double scale, int nc, V&& vals, double* out, bool* ok) {
  const auto& W  = GK::weights();                                        // Kronrod weights
  const auto& WG = boost::math::quadrature::gauss<double, (kNodes - 1) / 2>::weights();
  double K[kMaxComp], G[kMaxComp], fp[kMaxComp], fm[kMaxComp];
  int q = 0;
  // (kNodes - 1) / 2 = 30 is even (boost's else-branch): centre node with the
  // Kronrod weight only, then the Gauss nodes at the odd abscissa indices
  // (gauss_start = 1) and the Kronrod-only nodes at the even ones (kronrod_start = 2).
  vals(q++, fp);
  for(int c = 0; c < nc; c++) { K[c] = fp[c] * static_cast<double>(W[0]); G[c] = 0; }
  for(unsigned i = 1; i < W.size(); i += 2) {
    vals(q++, fp);
    vals(q++, fm);
    for(int c = 0; c < nc; c++) {
      K[c] += (fp[c] + fm[c]) * static_cast<double>(W[i]);
      G[c] += (fp[c] + fm[c]) * static_cast<double>(WG[i / 2]);
    }
  }
  for(unsigned i = 2; i < W.size(); i += 2) {
    vals(q++, fp);
    vals(q++, fm);
    for(int c = 0; c < nc; c++) K[c] += (fp[c] + fm[c]) * static_cast<double>(W[i]);
  }
  // recursive_adaptive_integrate, top level (abs_tol = 0, tol = root_epsilon):
  // error_local as integrate_non_adaptive_m1_1 forms it; subdivide iff
  // |estimate * tol| < error_local.
  const double tol = boost::math::tools::root_epsilon<double>();
  for(int c = 0; c < nc; c++) {
    const double err = (std::max)(static_cast<double>(std::abs(K[c] - G[c])),
                                  static_cast<double>(std::abs(K[c] * boost::math::tools::epsilon<double>() * double(2))));
    const double est = scale * K[c];
    const double tmp = est * tol;
    ok[c] = !(std::abs(tmp) < err);
    out[c] = est;
  }
}

// Bitwise equality of two doubles (the self-checks compare exact bit patterns).
inline bool sameBits(double x, double y) { return std::memcmp(&x, &y, sizeof(double)) == 0; }

// True when the translation unit that calls it was compiled with floating-point
// contraction, i.e. x * y + z may be evaluated as one fused multiply-add, within
// an expression (-ffp-contract=on) or across statements (-ffp-contract=fast); for
// example when the user's CXXFLAGS override the package's -ffp-contract=off.
// Contraction is then decided by the compiler case by case, so restructured code
// (the replica kernels, cached sums, cpwbart_grid's walks) need not round like the
// reference code, and the callers switch every shortcut off and keep the
// reference code path. The inputs are volatile so nothing is constant-folded:
// a * b = 1 + 2^-29 + 2^-60 exactly, which rounds to 1 + 2^-29, so both sums are
// exactly 0 unless the product is fused with the addition (then 2^-60). Also
// true where intermediate results carry excess precision (FLT_EVAL_METHOD != 0),
// which likewise breaks the replicas. 'static': each translation unit tests its
// own code generation.
static inline bool contractionDetected() {
  volatile double va = 1.0 + std::ldexp(1.0, -30);
  volatile double vc = -(1.0 + std::ldexp(1.0, -29));
  const double a = va, b = va, c = vc;
  const double r1 = a * b + c;   // within one expression
  const double p = a * b;
  const double r2 = p + c;       // across statements
  volatile double s1 = r1, s2 = r2;
  return s1 != 0.0 || s2 != 0.0;
}

} // namespace gk61

#endif
