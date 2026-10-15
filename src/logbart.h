// this part is for b-spline and random effect
// class should inherit bart class

#ifndef LOGBART_H
#define LOGBART_H

#include "bart.h"
#include "bd.h"

#include <ctime>
#include <vector>
// #include <eigen3/Eigen/Dense>
#include <RcppEigen.h>
#include <functional>
#include <random>

class logBART : public bart{
public:
  
  friend bool bd(tree& x, xinfo& xi, dinfo& di, pinfo& pi, double sigma,
                 std::vector<size_t>& nv, std::vector<double>& pv, bool aug, rn& gen,
                 double *fnoh, double *vij, double *ui, tree::tree_cp *leaf);
  
  logBART() : bart(), vij(NULL), ui(NULL) {}
  logBART(size_t m) : bart(m), vij(NULL), ui(NULL) {}
  ~logBART() {
    if(vij) delete[] vij;
    if(ui) delete[] ui;
  }
  
  
  void setdata(size_t p, size_t n, size_t nSubject, double *x, double *xObsTime, double *xIndex, double *y, size_t nc=100);
  //void setdata(size_t p, size_t n, double *x, double *y, int* nc);
  void setdata(size_t p, size_t n, size_t nSubject, double *x, double *xObsTime, double *xIndex, double *y, int* nc);
  
  void setwuv(const Rcpp::NumericVector& u, const Rcpp::NumericVector& v);
  void draw(double sigma, rn &gen);
  
protected:
  double *vij; // integral for exp(natural spline + random effect)
  double *ui;
};

void logBART::setwuv(const Rcpp::NumericVector& u, const Rcpp::NumericVector& v) {
  // Free existing memory if already allocated
  if (vij) delete[] vij;
  if (ui) delete[] ui;
  
  // Allocate new memory and copy data
  size_t n = u.size();
  vij = new double[n];
  ui = new double[n];
  
  std::memcpy(vij, &v[0], n * sizeof(double));
  std::memcpy(ui, &u[0], n * sizeof(double));
}




void logBART::draw(double sigma, rn &gen) {
  size_t i = 0;
  for(size_t j=0;j<m;j++) {
    //printf("tree j: %d\n", j);
    fit(t[j],xi,p,n,x,ftemp);
    for(size_t k=0;k<n;k++) {
      if(allfit[k] == 0 & ftemp[k] == 0){
        fnoh[k] = 1;
        allfit[k] = 1;
      }else{
        fnoh[k] = allfit[k] / ftemp[k];
        allfit[k] /= ftemp[k];
      }
      
      
      //printf("fnoh[%d]: %f\n", k, fnoh[k]);
      //printf("allfit[%d]: %f\n", k, allfit[k]);
      //printf("ftemp[%d]: %f\n", k, ftemp[k]);
    }
    if(bd(t[j],xi,di,pi,sigma,nv,pv,aug,gen,fnoh,vij,ui,0)) i++;
    //need to recode the drmu
    //printf("draw for all bottom nodes\n");
    drmu(t[j],xi,di,pi,sigma,gen,fnoh,vij,ui);
    fit(t[j],xi,p,n,x,ftemp);
    for(size_t k=0;k<n;k++) allfit[k] *= ftemp[k];
  }
  //print allfit
  for(size_t k=0;k<n;k++){
    printf("allfit[%d]: %f\n", k, allfit[k]);
  }
  
}




void logBART::setdata(size_t p, size_t n, size_t nSubject, double *x, double *xObsTime, double *xIndex, double *y, size_t numcut)
{
  int* nc = new int[p];
  for(size_t i=0; i<p; ++i) nc[i]=numcut;
  this->setdata(p,n,nSubject,x,xObsTime,xIndex,y,nc);
  delete [] nc;
}

void logBART::setdata(size_t p, size_t n, size_t nSubject, double *x, double *xObsTime, double *xIndex, double *y, int *nc)
{
  this->p=p; this->n=n;
  this->nSubject=nSubject;
  this->x=x; this->xObsTime=xObsTime; this->xIndex=xIndex;
  this->y=y;
  
  //transfer xIndex to is_final, if xIndex is 1,3,5 then the is_final is 0,1,0,1,0,1
  int* is_final = new int[n];
  for(size_t i=0; i<n; i++){
    //if i is in xIndex, then is_final[i] = 1
    is_final[i] = 0;
  }
  for(size_t i=1; i<=nSubject; i++){
    is_final[(int)xIndex[i]] = 1;
  }
  
  printf("Bart setdata\n");
  printf("p: %d\n", p);
  printf("n: %d\n", n);
  printf("nSubject: %d\n", nSubject);
  for(size_t i=0; i<n * p; i++){
    //printf("x: %f\n", *(x+i));
  }
  printf("xObsTime: %f\n", xObsTime[0]);
  printf("xObsTime: %f\n", xObsTime[n-1]);
  // for(size_t i=0; i<nSubject+1; i++){
  //   printf("xIndex: %f\n", xIndex[i]);
  // }
  printf("y: %f\n", y[0]);
  printf("y: %f\n", y[nSubject-1]);   // y (= y.train) has nSubject entries; y[n-1] read past its end
  for(size_t i=0; i<n; i++){
    //printf("is_final: %d\n", is_final[i]);
  }
  
  if(xi.size()==0) makexinfo(p,n,&x[0],xi,nc);
  
  if(allfit) delete[] allfit;
  allfit = new double[n];
  predict(p,n,x,allfit);
  //print allfit
  // for(size_t i=0; i<n; i++){
  //   printf("allfit: %f\n", allfit[i]);
  // }
  
  if(r) delete[] r;
  r = new double[n];
  
  if(ftemp) delete[] ftemp;
  ftemp = new double[n];
  
  if(fnoh) delete[] fnoh;
  fnoh = new double[n];
  //
  if(nodeid) delete[] nodeid;
  nodeid = new double[n];
  
  di.n=n; di.nSubject = nSubject; di.p=p; di.x = &x[0]; di.y=r;
  di.xObsTime = xObsTime; di.xIndex = xIndex; di.is_final = is_final;
  for(size_t j=0;j<p;j++){
    nv.push_back(0);
    pv.push_back(1/(double)p);
  }
}






#endif
