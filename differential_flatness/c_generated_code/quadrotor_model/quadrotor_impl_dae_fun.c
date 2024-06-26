/* This file was automatically generated by CasADi 3.6.5.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) quadrotor_impl_dae_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_house CASADI_PREFIX(house)
#define casadi_if_else CASADI_PREFIX(if_else)
#define casadi_qr CASADI_PREFIX(qr)
#define casadi_qr_colcomb CASADI_PREFIX(qr_colcomb)
#define casadi_qr_mv CASADI_PREFIX(qr_mv)
#define casadi_qr_singular CASADI_PREFIX(qr_singular)
#define casadi_qr_solve CASADI_PREFIX(qr_solve)
#define casadi_qr_trs CASADI_PREFIX(qr_trs)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_scal CASADI_PREFIX(scal)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_sq(casadi_real x) { return x*x;}

casadi_real casadi_if_else(casadi_real c, casadi_real x, casadi_real y) { return c!=0 ? x : y;}

void casadi_scal(casadi_int n, casadi_real alpha, casadi_real* x) {
  casadi_int i;
  if (!x) return;
  for (i=0; i<n; ++i) *x++ *= alpha;
}

casadi_real casadi_dot(casadi_int n, const casadi_real* x, const casadi_real* y) {
  casadi_int i;
  casadi_real r = 0;
  for (i=0; i<n; ++i) r += *x++ * *y++;
  return r;
}

casadi_real casadi_house(casadi_real* v, casadi_real* beta, casadi_int nv) {
  casadi_int i;
  casadi_real v0, sigma, s, sigma_is_zero, v0_nonpos;
  v0 = v[0];
  sigma=0;
  for (i=1; i<nv; ++i) sigma += v[i]*v[i];
  s = sqrt(v0*v0 + sigma);
  sigma_is_zero = sigma==0;
  v0_nonpos = v0<=0;
  v[0] = casadi_if_else(sigma_is_zero, 1,
                 casadi_if_else(v0_nonpos, v0-s, -sigma/(v0+s)));
  *beta = casadi_if_else(sigma_is_zero, 2*v0_nonpos, -1/(s*v[0]));
  return s;
}
void casadi_qr(const casadi_int* sp_a, const casadi_real* nz_a, casadi_real* x,
               const casadi_int* sp_v, casadi_real* nz_v, const casadi_int* sp_r, casadi_real* nz_r, casadi_real* beta,
               const casadi_int* prinv, const casadi_int* pc) {
   casadi_int ncol, nrow, r, c, k, k1;
   casadi_real alpha;
   const casadi_int *a_colind, *a_row, *v_colind, *v_row, *r_colind, *r_row;
   ncol = sp_a[1];
   a_colind=sp_a+2; a_row=sp_a+2+ncol+1;
   nrow = sp_v[0];
   v_colind=sp_v+2; v_row=sp_v+2+ncol+1;
   r_colind=sp_r+2; r_row=sp_r+2+ncol+1;
   for (r=0; r<nrow; ++r) x[r] = 0;
   for (c=0; c<ncol; ++c) {
     for (k=a_colind[pc[c]]; k<a_colind[pc[c]+1]; ++k) x[prinv[a_row[k]]] = nz_a[k];
     for (k=r_colind[c]; k<r_colind[c+1] && (r=r_row[k])<c; ++k) {
       alpha = 0;
       for (k1=v_colind[r]; k1<v_colind[r+1]; ++k1) alpha += nz_v[k1]*x[v_row[k1]];
       alpha *= beta[r];
       for (k1=v_colind[r]; k1<v_colind[r+1]; ++k1) x[v_row[k1]] -= alpha*nz_v[k1];
       *nz_r++ = x[r];
       x[r] = 0;
     }
     for (k=v_colind[c]; k<v_colind[c+1]; ++k) {
       nz_v[k] = x[v_row[k]];
       x[v_row[k]] = 0;
     }
     *nz_r++ = casadi_house(nz_v + v_colind[c], beta + c, v_colind[c+1] - v_colind[c]);
   }
 }
void casadi_qr_mv(const casadi_int* sp_v, const casadi_real* v, const casadi_real* beta, casadi_real* x,
                  casadi_int tr) {
  casadi_int ncol, c, c1, k;
  casadi_real alpha;
  const casadi_int *colind, *row;
  ncol=sp_v[1];
  colind=sp_v+2; row=sp_v+2+ncol+1;
  for (c1=0; c1<ncol; ++c1) {
    c = tr ? c1 : ncol-1-c1;
    alpha=0;
    for (k=colind[c]; k<colind[c+1]; ++k) alpha += v[k]*x[row[k]];
    alpha *= beta[c];
    for (k=colind[c]; k<colind[c+1]; ++k) x[row[k]] -= alpha*v[k];
  }
}
void casadi_qr_trs(const casadi_int* sp_r, const casadi_real* nz_r, casadi_real* x, casadi_int tr) {
  casadi_int ncol, r, c, k;
  const casadi_int *colind, *row;
  ncol=sp_r[1];
  colind=sp_r+2; row=sp_r+2+ncol+1;
  if (tr) {
    for (c=0; c<ncol; ++c) {
      for (k=colind[c]; k<colind[c+1]; ++k) {
        r = row[k];
        if (r==c) {
          x[c] /= nz_r[k];
        } else {
          x[c] -= nz_r[k]*x[r];
        }
      }
    }
  } else {
    for (c=ncol-1; c>=0; --c) {
      for (k=colind[c+1]-1; k>=colind[c]; --k) {
        r=row[k];
        if (r==c) {
          x[r] /= nz_r[k];
        } else {
          x[r] -= nz_r[k]*x[c];
        }
      }
    }
  }
}
void casadi_qr_solve(casadi_real* x, casadi_int nrhs, casadi_int tr,
                     const casadi_int* sp_v, const casadi_real* v, const casadi_int* sp_r, const casadi_real* r,
                     const casadi_real* beta, const casadi_int* prinv, const casadi_int* pc, casadi_real* w) {
  casadi_int k, c, nrow_ext, ncol;
  nrow_ext = sp_v[0]; ncol = sp_v[1];
  for (k=0; k<nrhs; ++k) {
    if (tr) {
      for (c=0; c<ncol; ++c) w[c] = x[pc[c]];
      casadi_qr_trs(sp_r, r, w, 1);
      casadi_qr_mv(sp_v, v, beta, w, 0);
      for (c=0; c<ncol; ++c) x[c] = w[prinv[c]];
    } else {
      for (c=0; c<nrow_ext; ++c) w[c] = 0;
      for (c=0; c<ncol; ++c) w[prinv[c]] = x[c];
      casadi_qr_mv(sp_v, v, beta, w, 1);
      casadi_qr_trs(sp_r, r, w, 0);
      for (c=0; c<ncol; ++c) x[pc[c]] = w[c];
    }
    x += ncol;
  }
}
casadi_int casadi_qr_singular(casadi_real* rmin, casadi_int* irmin, const casadi_real* nz_r,
                             const casadi_int* sp_r, const casadi_int* pc, casadi_real eps) {
  casadi_real rd, rd_min;
  casadi_int ncol, c, nullity;
  const casadi_int* r_colind;
  nullity = 0;
  ncol = sp_r[1];
  r_colind = sp_r + 2;
  for (c=0; c<ncol; ++c) {
    rd = fabs(nz_r[r_colind[c+1]-1]);
    if (rd<eps) nullity++;
    if (c==0 || rd < rd_min) {
      rd_min = rd;
      if (rmin) *rmin = rd;
      if (irmin) *irmin = pc[c];
    }
  }
  return nullity;
}
void casadi_qr_colcomb(casadi_real* v, const casadi_real* nz_r, const casadi_int* sp_r,
                       const casadi_int* pc, casadi_real eps, casadi_int ind) {
  casadi_int ncol, r, c, k;
  const casadi_int *r_colind, *r_row;
  ncol = sp_r[1];
  r_colind = sp_r + 2;
  r_row = r_colind + ncol + 1;
  for (c=0; c<ncol; ++c) {
    if (fabs(nz_r[r_colind[c+1]-1])<eps && 0==ind--) {
      ind = c;
      break;
    }
  }
  casadi_clear(v, ncol);
  v[pc[ind]] = 1.;
  for (k=r_colind[ind]; k<r_colind[ind+1]-1; ++k) {
    v[pc[r_row[k]]] = -nz_r[k];
  }
  for (c=ind-1; c>=0; --c) {
    for (k=r_colind[c+1]-1; k>=r_colind[c]; --k) {
      r=r_row[k];
      if (r==c) {
        if (fabs(nz_r[k])<eps) {
          v[pc[r]] = 0;
        } else {
          v[pc[r]] /= nz_r[k];
        }
      } else {
        v[pc[r]] -= nz_r[k]*v[pc[c]];
      }
    }
  }
  casadi_scal(ncol, 1./sqrt(casadi_dot(ncol, v, v)), v);
}

static const casadi_int casadi_s0[3] = {0, 1, 2};
static const casadi_int casadi_s1[15] = {3, 3, 0, 3, 6, 9, 0, 1, 2, 0, 1, 2, 0, 1, 2};
static const casadi_int casadi_s2[12] = {3, 3, 0, 3, 5, 6, 0, 1, 2, 1, 2, 2};
static const casadi_int casadi_s3[12] = {3, 3, 0, 1, 3, 6, 0, 0, 1, 0, 1, 2};
static const casadi_int casadi_s4[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s5[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s6[3] = {0, 0, 0};
static const casadi_int casadi_s7[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

static const casadi_real casadi_c0[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};

/* quadrotor_impl_dae_fun:(i0[13],i1[13],i2[4],i3[],i4[],i5[17])->(o0[13]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cr, *cs;
  casadi_real w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, *w13=w+17, *w14=w+30, *w15=w+43, *w16=w+46, *w17=w+49, *w18=w+53, w19, w20, w21, *w22=w+60, *w23=w+63, *w24=w+66, *w25=w+69, *w26=w+78, *w27=w+87, *w28=w+91, *w29=w+95, *w30=w+99, *w31=w+103, *w32=w+119, *w33=w+135;
  /* #0: @0 = input[1][0] */
  w0 = arg[1] ? arg[1][0] : 0;
  /* #1: @1 = input[1][1] */
  w1 = arg[1] ? arg[1][1] : 0;
  /* #2: @2 = input[1][2] */
  w2 = arg[1] ? arg[1][2] : 0;
  /* #3: @3 = input[1][3] */
  w3 = arg[1] ? arg[1][3] : 0;
  /* #4: @4 = input[1][4] */
  w4 = arg[1] ? arg[1][4] : 0;
  /* #5: @5 = input[1][5] */
  w5 = arg[1] ? arg[1][5] : 0;
  /* #6: @6 = input[1][6] */
  w6 = arg[1] ? arg[1][6] : 0;
  /* #7: @7 = input[1][7] */
  w7 = arg[1] ? arg[1][7] : 0;
  /* #8: @8 = input[1][8] */
  w8 = arg[1] ? arg[1][8] : 0;
  /* #9: @9 = input[1][9] */
  w9 = arg[1] ? arg[1][9] : 0;
  /* #10: @10 = input[1][10] */
  w10 = arg[1] ? arg[1][10] : 0;
  /* #11: @11 = input[1][11] */
  w11 = arg[1] ? arg[1][11] : 0;
  /* #12: @12 = input[1][12] */
  w12 = arg[1] ? arg[1][12] : 0;
  /* #13: @13 = vertcat(@0, @1, @2, @3, @4, @5, @6, @7, @8, @9, @10, @11, @12) */
  rr=w13;
  *rr++ = w0;
  *rr++ = w1;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w12;
  /* #14: @0 = input[0][0] */
  w0 = arg[0] ? arg[0][0] : 0;
  /* #15: @1 = input[0][1] */
  w1 = arg[0] ? arg[0][1] : 0;
  /* #16: @2 = input[0][2] */
  w2 = arg[0] ? arg[0][2] : 0;
  /* #17: @3 = input[0][3] */
  w3 = arg[0] ? arg[0][3] : 0;
  /* #18: @4 = input[0][4] */
  w4 = arg[0] ? arg[0][4] : 0;
  /* #19: @5 = input[0][5] */
  w5 = arg[0] ? arg[0][5] : 0;
  /* #20: @6 = input[0][6] */
  w6 = arg[0] ? arg[0][6] : 0;
  /* #21: @7 = input[0][7] */
  w7 = arg[0] ? arg[0][7] : 0;
  /* #22: @8 = input[0][8] */
  w8 = arg[0] ? arg[0][8] : 0;
  /* #23: @9 = input[0][9] */
  w9 = arg[0] ? arg[0][9] : 0;
  /* #24: @10 = input[0][10] */
  w10 = arg[0] ? arg[0][10] : 0;
  /* #25: @11 = input[0][11] */
  w11 = arg[0] ? arg[0][11] : 0;
  /* #26: @12 = input[0][12] */
  w12 = arg[0] ? arg[0][12] : 0;
  /* #27: @14 = vertcat(@0, @1, @2, @3, @4, @5, @6, @7, @8, @9, @10, @11, @12) */
  rr=w14;
  *rr++ = w0;
  *rr++ = w1;
  *rr++ = w2;
  *rr++ = w3;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w12;
  /* #28: @15 = @14[3:6] */
  for (rr=w15, ss=w14+3; ss!=w14+6; ss+=1) *rr++ = *ss;
  /* #29: @0 = input[2][0] */
  w0 = arg[2] ? arg[2][0] : 0;
  /* #30: @16 = zeros(3x1) */
  casadi_clear(w16, 3);
  /* #31: @17 = @14[6:10] */
  for (rr=w17, ss=w14+6; ss!=w14+10; ss+=1) *rr++ = *ss;
  /* #32: @1 = 0 */
  w1 = 0.;
  /* #33: @18 = @17' */
  casadi_copy(w17, 4, w18);
  /* #34: @1 = mac(@18,@17,@1) */
  for (i=0, rr=(&w1); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w18+j, tt=w17+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #35: @18 = (@17/@1) */
  for (i=0, rr=w18, cr=w17; i<4; ++i) (*rr++)  = ((*cr++)/w1);
  /* #36: @1 = @18[0] */
  for (rr=(&w1), ss=w18+0; ss!=w18+1; ss+=1) *rr++ = *ss;
  /* #37: @2 = sq(@1) */
  w2 = casadi_sq( w1 );
  /* #38: @3 = @18[1] */
  for (rr=(&w3), ss=w18+1; ss!=w18+2; ss+=1) *rr++ = *ss;
  /* #39: @4 = sq(@3) */
  w4 = casadi_sq( w3 );
  /* #40: @2 = (@2+@4) */
  w2 += w4;
  /* #41: @4 = @18[2] */
  for (rr=(&w4), ss=w18+2; ss!=w18+3; ss+=1) *rr++ = *ss;
  /* #42: @5 = sq(@4) */
  w5 = casadi_sq( w4 );
  /* #43: @2 = (@2-@5) */
  w2 -= w5;
  /* #44: @5 = @18[3] */
  for (rr=(&w5), ss=w18+3; ss!=w18+4; ss+=1) *rr++ = *ss;
  /* #45: @19 = sq(@5) */
  w19 = casadi_sq( w5 );
  /* #46: @2 = (@2-@19) */
  w2 -= w19;
  /* #47: @19 = (@3*@4) */
  w19  = (w3*w4);
  /* #48: @20 = (@1*@5) */
  w20  = (w1*w5);
  /* #49: @19 = (@19-@20) */
  w19 -= w20;
  /* #50: @19 = (2.*@19) */
  w19 = (2.* w19 );
  /* #51: @20 = (@3*@5) */
  w20  = (w3*w5);
  /* #52: @21 = (@1*@4) */
  w21  = (w1*w4);
  /* #53: @20 = (@20+@21) */
  w20 += w21;
  /* #54: @20 = (2.*@20) */
  w20 = (2.* w20 );
  /* #55: @22 = horzcat(@2, @19, @20) */
  rr=w22;
  *rr++ = w2;
  *rr++ = w19;
  *rr++ = w20;
  /* #56: @22 = @22' */
  /* #57: @2 = (@3*@4) */
  w2  = (w3*w4);
  /* #58: @19 = (@1*@5) */
  w19  = (w1*w5);
  /* #59: @2 = (@2+@19) */
  w2 += w19;
  /* #60: @2 = (2.*@2) */
  w2 = (2.* w2 );
  /* #61: @19 = sq(@1) */
  w19 = casadi_sq( w1 );
  /* #62: @20 = sq(@4) */
  w20 = casadi_sq( w4 );
  /* #63: @19 = (@19+@20) */
  w19 += w20;
  /* #64: @20 = sq(@3) */
  w20 = casadi_sq( w3 );
  /* #65: @19 = (@19-@20) */
  w19 -= w20;
  /* #66: @20 = sq(@5) */
  w20 = casadi_sq( w5 );
  /* #67: @19 = (@19-@20) */
  w19 -= w20;
  /* #68: @20 = (@4*@5) */
  w20  = (w4*w5);
  /* #69: @21 = (@1*@3) */
  w21  = (w1*w3);
  /* #70: @20 = (@20-@21) */
  w20 -= w21;
  /* #71: @20 = (2.*@20) */
  w20 = (2.* w20 );
  /* #72: @23 = horzcat(@2, @19, @20) */
  rr=w23;
  *rr++ = w2;
  *rr++ = w19;
  *rr++ = w20;
  /* #73: @23 = @23' */
  /* #74: @2 = (@3*@5) */
  w2  = (w3*w5);
  /* #75: @19 = (@1*@4) */
  w19  = (w1*w4);
  /* #76: @2 = (@2-@19) */
  w2 -= w19;
  /* #77: @2 = (2.*@2) */
  w2 = (2.* w2 );
  /* #78: @19 = (@4*@5) */
  w19  = (w4*w5);
  /* #79: @20 = (@1*@3) */
  w20  = (w1*w3);
  /* #80: @19 = (@19+@20) */
  w19 += w20;
  /* #81: @19 = (2.*@19) */
  w19 = (2.* w19 );
  /* #82: @1 = sq(@1) */
  w1 = casadi_sq( w1 );
  /* #83: @5 = sq(@5) */
  w5 = casadi_sq( w5 );
  /* #84: @1 = (@1+@5) */
  w1 += w5;
  /* #85: @3 = sq(@3) */
  w3 = casadi_sq( w3 );
  /* #86: @1 = (@1-@3) */
  w1 -= w3;
  /* #87: @4 = sq(@4) */
  w4 = casadi_sq( w4 );
  /* #88: @1 = (@1-@4) */
  w1 -= w4;
  /* #89: @24 = horzcat(@2, @19, @1) */
  rr=w24;
  *rr++ = w2;
  *rr++ = w19;
  *rr++ = w1;
  /* #90: @24 = @24' */
  /* #91: @25 = horzcat(@22, @23, @24) */
  rr=w25;
  for (i=0, cs=w22; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w23; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w24; i<3; ++i) *rr++ = *cs++;
  /* #92: @26 = @25' */
  for (i=0, rr=w26, cs=w25; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #93: @22 = zeros(3x1) */
  casadi_clear(w22, 3);
  /* #94: @2 = 1 */
  w2 = 1.;
  /* #95: (@22[2] = @2) */
  for (rr=w22+2, ss=(&w2); rr!=w22+3; rr+=1) *rr = *ss++;
  /* #96: @16 = mac(@26,@22,@16) */
  for (i=0, rr=w16; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w26+j, tt=w22+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #97: @16 = (@0*@16) */
  for (i=0, rr=w16, cs=w16; i<3; ++i) (*rr++)  = (w0*(*cs++));
  /* #98: @2 = 9.8 */
  w2 = 9.8000000000000007e+00;
  /* #99: @22 = (@2*@22) */
  for (i=0, rr=w22, cs=w22; i<3; ++i) (*rr++)  = (w2*(*cs++));
  /* #100: @16 = (@16-@22) */
  for (i=0, rr=w16, cs=w22; i<3; ++i) (*rr++) -= (*cs++);
  /* #101: @2 = 0.5 */
  w2 = 5.0000000000000000e-01;
  /* #102: @18 = zeros(4x1) */
  casadi_clear(w18, 4);
  /* #103: @19 = (-@7) */
  w19 = (- w7 );
  /* #104: @1 = (-@8) */
  w1 = (- w8 );
  /* #105: @4 = (-@9) */
  w4 = (- w9 );
  /* #106: @27 = horzcat(@6, @19, @1, @4) */
  rr=w27;
  *rr++ = w6;
  *rr++ = w19;
  *rr++ = w1;
  *rr++ = w4;
  /* #107: @27 = @27' */
  /* #108: @19 = (-@9) */
  w19 = (- w9 );
  /* #109: @28 = horzcat(@7, @6, @19, @8) */
  rr=w28;
  *rr++ = w7;
  *rr++ = w6;
  *rr++ = w19;
  *rr++ = w8;
  /* #110: @28 = @28' */
  /* #111: @19 = (-@7) */
  w19 = (- w7 );
  /* #112: @29 = horzcat(@8, @9, @6, @19) */
  rr=w29;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w6;
  *rr++ = w19;
  /* #113: @29 = @29' */
  /* #114: @19 = (-@8) */
  w19 = (- w8 );
  /* #115: @30 = horzcat(@9, @19, @7, @6) */
  rr=w30;
  *rr++ = w9;
  *rr++ = w19;
  *rr++ = w7;
  *rr++ = w6;
  /* #116: @30 = @30' */
  /* #117: @31 = horzcat(@27, @28, @29, @30) */
  rr=w31;
  for (i=0, cs=w27; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w28; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w29; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w30; i<4; ++i) *rr++ = *cs++;
  /* #118: @32 = @31' */
  for (i=0, rr=w32, cs=w31; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #119: @19 = 0 */
  w19 = 0.;
  /* #120: @27 = vertcat(@19, @10, @11, @12) */
  rr=w27;
  *rr++ = w19;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w12;
  /* #121: @18 = mac(@32,@27,@18) */
  for (i=0, rr=w18; i<1; ++i) for (j=0; j<4; ++j, ++rr) for (k=0, ss=w32+j, tt=w27+i*4; k<4; ++k) *rr += ss[k*4]**tt++;
  /* #122: @18 = (@2*@18) */
  for (i=0, rr=w18, cs=w18; i<4; ++i) (*rr++)  = (w2*(*cs++));
  /* #123: @2 = 10 */
  w2 = 10.;
  /* #124: @19 = 1 */
  w19 = 1.;
  /* #125: @6 = sq(@6) */
  w6 = casadi_sq( w6 );
  /* #126: @7 = sq(@7) */
  w7 = casadi_sq( w7 );
  /* #127: @6 = (@6+@7) */
  w6 += w7;
  /* #128: @8 = sq(@8) */
  w8 = casadi_sq( w8 );
  /* #129: @6 = (@6+@8) */
  w6 += w8;
  /* #130: @9 = sq(@9) */
  w9 = casadi_sq( w9 );
  /* #131: @6 = (@6+@9) */
  w6 += w9;
  /* #132: @19 = (@19-@6) */
  w19 -= w6;
  /* #133: @2 = (@2*@19) */
  w2 *= w19;
  /* #134: @17 = (@2*@17) */
  for (i=0, rr=w17, cs=w17; i<4; ++i) (*rr++)  = (w2*(*cs++));
  /* #135: @18 = (@18+@17) */
  for (i=0, rr=w18, cs=w17; i<4; ++i) (*rr++) += (*cs++);
  /* #136: @22 = zeros(3x1) */
  casadi_clear(w22, 3);
  /* #137: @26 = 
  [[1, 0, 0], 
   [0, 1, 0], 
   [0, 0, 1]] */
  casadi_copy(casadi_c0, 9, w26);
  /* #138: @25 = zeros(3x3) */
  casadi_clear(w25, 9);
  /* #139: @2 = 0.00264 */
  w2 = 2.6400000000000000e-03;
  /* #140: (@25[0] = @2) */
  for (rr=w25+0, ss=(&w2); rr!=w25+1; rr+=1) *rr = *ss++;
  /* #141: @2 = 0.00264 */
  w2 = 2.6400000000000000e-03;
  /* #142: (@25[4] = @2) */
  for (rr=w25+4, ss=(&w2); rr!=w25+5; rr+=1) *rr = *ss++;
  /* #143: @2 = 0.00496 */
  w2 = 4.9600000000000000e-03;
  /* #144: (@25[8] = @2) */
  for (rr=w25+8, ss=(&w2); rr!=w25+9; rr+=1) *rr = *ss++;
  /* #145: @26 = (@25\@26) */
  rr = w26;
  ss = w25;
  {
    /* FIXME(@jaeandersson): Memory allocation can be avoided */
    casadi_real v[6], r[6], beta[3], w[6];
    casadi_qr(casadi_s1, ss, w, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0);
    casadi_qr_solve(rr, 3, 0, casadi_s2, v, casadi_s3, r, beta, casadi_s0, casadi_s0, w);
  }
  /* #146: @2 = input[2][1] */
  w2 = arg[2] ? arg[2][1] : 0;
  /* #147: @19 = input[2][2] */
  w19 = arg[2] ? arg[2][2] : 0;
  /* #148: @6 = input[2][3] */
  w6 = arg[2] ? arg[2][3] : 0;
  /* #149: @17 = vertcat(@0, @2, @19, @6) */
  rr=w17;
  *rr++ = w0;
  *rr++ = w2;
  *rr++ = w19;
  *rr++ = w6;
  /* #150: @23 = @17[1:4] */
  for (rr=w23, ss=w17+1; ss!=w17+4; ss+=1) *rr++ = *ss;
  /* #151: @24 = zeros(3x1) */
  casadi_clear(w24, 3);
  /* #152: @33 = @14[10:13] */
  for (rr=w33, ss=w14+10; ss!=w14+13; ss+=1) *rr++ = *ss;
  /* #153: @24 = mac(@25,@33,@24) */
  for (i=0, rr=w24; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w25+j, tt=w33+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #154: @0 = @24[2] */
  for (rr=(&w0), ss=w24+2; ss!=w24+3; ss+=1) *rr++ = *ss;
  /* #155: @2 = (@11*@0) */
  w2  = (w11*w0);
  /* #156: @19 = @24[1] */
  for (rr=(&w19), ss=w24+1; ss!=w24+2; ss+=1) *rr++ = *ss;
  /* #157: @6 = (@12*@19) */
  w6  = (w12*w19);
  /* #158: @2 = (@2-@6) */
  w2 -= w6;
  /* #159: @6 = @24[0] */
  for (rr=(&w6), ss=w24+0; ss!=w24+1; ss+=1) *rr++ = *ss;
  /* #160: @12 = (@12*@6) */
  w12 *= w6;
  /* #161: @0 = (@10*@0) */
  w0  = (w10*w0);
  /* #162: @12 = (@12-@0) */
  w12 -= w0;
  /* #163: @10 = (@10*@19) */
  w10 *= w19;
  /* #164: @11 = (@11*@6) */
  w11 *= w6;
  /* #165: @10 = (@10-@11) */
  w10 -= w11;
  /* #166: @24 = vertcat(@2, @12, @10) */
  rr=w24;
  *rr++ = w2;
  *rr++ = w12;
  *rr++ = w10;
  /* #167: @23 = (@23-@24) */
  for (i=0, rr=w23, cs=w24; i<3; ++i) (*rr++) -= (*cs++);
  /* #168: @22 = mac(@26,@23,@22) */
  for (i=0, rr=w22; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w26+j, tt=w23+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #169: @14 = vertcat(@15, @16, @18, @22) */
  rr=w14;
  for (i=0, cs=w15; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w16; i<3; ++i) *rr++ = *cs++;
  for (i=0, cs=w18; i<4; ++i) *rr++ = *cs++;
  for (i=0, cs=w22; i<3; ++i) *rr++ = *cs++;
  /* #170: @13 = (@13-@14) */
  for (i=0, rr=w13, cs=w14; i<13; ++i) (*rr++) -= (*cs++);
  /* #171: output[0][0] = @13 */
  casadi_copy(w13, 13, res[0]);
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_impl_dae_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int quadrotor_impl_dae_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_impl_dae_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quadrotor_impl_dae_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int quadrotor_impl_dae_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void quadrotor_impl_dae_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void quadrotor_impl_dae_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void quadrotor_impl_dae_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int quadrotor_impl_dae_fun_n_in(void) { return 6;}

CASADI_SYMBOL_EXPORT casadi_int quadrotor_impl_dae_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real quadrotor_impl_dae_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quadrotor_impl_dae_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    case 5: return "i5";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* quadrotor_impl_dae_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quadrotor_impl_dae_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    case 3: return casadi_s6;
    case 4: return casadi_s6;
    case 5: return casadi_s7;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* quadrotor_impl_dae_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int quadrotor_impl_dae_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 19;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 138;
  return 0;
}

CASADI_SYMBOL_EXPORT int quadrotor_impl_dae_fun_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 19*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 2*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 138*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
