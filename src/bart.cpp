/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017 Robert McCulloch and Rodney Sparapani
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, a copy is available at
 *  https://www.R-project.org/Licenses/GPL-2
 */

#include "bart.h"

//--------------------------------------------------
//constructor
bart::bart():m(200),t(m),pi(),p(0),n(0),x(0),y(0),xi(),allfit(0),r(0),ftemp(0),nodeid(0),fnoh(0),di(),dartOn(false),accept(0.),grp(0), aug(0) {}
bart::bart(size_t im):m(im),t(m),pi(),p(0),n(0),x(0),y(0),xi(),allfit(0),r(0),ftemp(0),nodeid(0),fnoh(0),di(),dartOn(false),accept(0.),grp(0), aug(0) {}
bart::bart(const bart& ib):m(ib.m),t(m),pi(ib.pi),p(0),n(0),x(0),y(0),xi(),allfit(0),r(0),ftemp(0),nodeid(0),fnoh(0),di(),dartOn(false),accept(0.),grp(0), aug(0)
{
  this->t = ib.t;
}
bart::~bart()
{
  if(allfit) delete[] allfit;
  if(r) delete[] r;
  if(ftemp) delete[] ftemp;
  if(grp) delete[] grp;
  if(nodeid) delete[] nodeid;
  if(fnoh) delete[] fnoh;
}

//--------------------------------------------------
//operators
bart& bart::operator=(const bart& rhs)
{
  
  printf("Bart operator=\n");
  if(&rhs != this) {
    
    this->t = rhs.t;
    this->m = t.size();
    
    this->pi = rhs.pi;
    
    p=0;n=0;x=0;y=0;
    xi.clear();
    
    if(allfit) {delete[] allfit; allfit=0;}
    if(r) {delete[] r; r=0;}
    if(ftemp) {delete[] ftemp; ftemp=0;}
    
  }
  return *this;
}
//--------------------------------------------------
//get,set
void bart::setm(size_t m)
{
  t.resize(m);
  this->m = t.size();
  
  if(allfit && (xi.size()==p)) predict(p,n,x,allfit);
}

//--------------------------------------------------
void bart::setxinfo(xinfo& _xi)
{
  size_t p=_xi.size();
  xi.resize(p);
  for(size_t i=0;i<p;i++) {
    size_t nc=_xi[i].size();
    xi[i].resize(nc);
    for(size_t j=0;j<nc;j++) xi[i][j] = _xi[i][j];
  }
}
//--------------------------------------------------
void bart::setdata(size_t p, size_t n, size_t nSubject, double *x, double *xObsTime, double *xIndex, double *xEvent, double *y, size_t numcut)
{
  int* nc = new int[p];
  for(size_t i=0; i<p; ++i) nc[i]=numcut;
  this->setdata(p,n,nSubject,x,xObsTime,xIndex,xEvent,y,nc);
  delete [] nc;
}

void bart::setdata(size_t p, size_t n, size_t nSubject, double *x, double *xObsTime, double *xIndex, double *xEvent, double *y, int *nc)
{
  this->p=p; this->n=n;
  this->nSubject=nSubject;
  this->x=x; this->xObsTime=xObsTime; this->xIndex=xIndex; this->xEvent=xEvent;
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
    printf("x: %f\n", *(x+i));
  }
  printf("xObsTime: %f\n", xObsTime[0]);
  printf("xObsTime: %f\n", xObsTime[n-1]);
  for(size_t i=0; i<nSubject+1; i++){
    printf("xIndex: %f\n", xIndex[i]);
  }
  printf("xEvent: %f\n", xEvent[0]);
  printf("xEvent: %f\n", xEvent[n-1]);
  printf("y: %f\n", y[0]);
  printf("y: %f\n", y[nSubject-1]);   // y has nSubject entries (bart::setdata is not called by cwbart)
  for(size_t i=0; i<n; i++){
    printf("is_final: %d\n", is_final[i]);
  }
  
  if(xi.size()==0) makexinfo(p,n,&x[0],xi,nc);
  
  if(allfit) delete[] allfit;
  allfit = new double[n];
  predict(p,n,x,allfit);
  
  
  if(r) delete[] r;
  r = new double[n];
  
  if(ftemp) delete[] ftemp;
  ftemp = new double[n];
  for(size_t j=0;j<n;j++) ftemp[j]=1.0;
  
  if(fnoh) delete[] fnoh;
  fnoh = new double[n];
  for(size_t j=0;j<n;j++) fnoh[j]=1.0;

  if(nodeid) delete[] nodeid;
  nodeid = new double[n];
  
  di.n=n; di.nSubject = nSubject; di.p=p; di.x = &x[0]; di.y=r;
  di.xObsTime = xObsTime; di.xIndex = xIndex; di.xEvent = xEvent; di.is_final = is_final;
  
  //save to struct nutsData save double * to Eigen::VectorXd
  nutsData.xObsTime = Eigen::VectorXd::Zero(n);
  nutsData.xIndex = Eigen::VectorXd::Zero(nSubject + 1);
  nutsData.xEvent = Eigen::VectorXd::Zero(n);
  nutsData.y = Eigen::VectorXd::Zero(nSubject);
  nutsData.ReSubjectId = -1;
  
  for(size_t i=0; i<nSubject; i++){
    nutsData.xIndex[i] = xIndex[i];
    nutsData.y[i] = y[i];
  }
  nutsData.xIndex[nSubject] = xIndex[nSubject]; //last index is the end of the last subject
  for(size_t i=0; i<n; i++){
    nutsData.xObsTime[i] = xObsTime[i];
    nutsData.xEvent[i] = xEvent[i];
  }
  
  
  for(size_t j=0;j<p;j++){
    nv.push_back(0);
    pv.push_back(1/(double)p);
  }
}
//--------------------------------------------------
void bart::predict(size_t p, size_t n, double *x, double *fp)
  //uses: m,t,xi
{
  double *fptemp = new double[n];
  
  //initailize fp
  
  for(size_t j=0;j<n;j++) fp[j]=1.0;
  for(size_t j=0;j<m;j++) {
    fit(t[j],xi,p,n,x,fptemp);
    // code for log linear BART
    //for(size_t k=0;k<n;k++) fp[k] *= fptemp[k];
    for(size_t k=0;k<n;k++) {
      //printf("fptemp: %f\n", fptemp[k]);
      fp[k] *= fptemp[k];
    }
  }
  
  delete[] fptemp;
}
//--------------------------------------------------
// void bart::draw(double sigma, rn& gen)
// {
//    size_t i=0;
//    for(size_t j=0;j<m;j++) {
//       fit(t[j],xi,p,n,x,ftemp,nodeid);
//       for(size_t k=0;k<n;k++) {
//         allfit[k] = allfit[k]-ftemp[k];
//         r[k] = y[k]-allfit[k];
//         fnoh[k] = k;
//       }
//       if(bd(t[j],xi,di,pi,sigma,nv,pv,aug,gen,fnoh,nodeid)) i++;
//       drmu(t[j],xi,di,pi,sigma,gen);
//       fit(t[j],xi,p,n,x,ftemp);
//       for(size_t k=0;k<n;k++) allfit[k] += ftemp[k];
//    }
//    accept=i/(double)m;
//    if(dartOn) {
//      if(grp) draw_s_grp(nv,lpv,theta,gen,grp,rho);
//      else draw_s(nv,lpv,theta,gen);
//      draw_theta0(const_theta,theta,lpv,a,b,rho,gen);
//      for(size_t j=0;j<p;j++) pv[j]=::exp(lpv[j]);
//    }
// }
//--------------------------------------------------
//public functions
void bart::pr() //print to screen
{
  cout << "*****bart object:\n";
  cout << "m: " << m << std::endl;
  cout << "t[0]:\n " << t[0] << std::endl;
  cout << "t[m-1]:\n " << t[m-1] << std::endl;
  cout << "prior and mcmc info:\n";
  pi.pr();
  if(dart){
    cout << "*****dart prior (On):\n";
    cout << "a: " << a << std::endl;
    cout << "b: " << b << std::endl;
    cout << "rho: " << rho << std::endl;
    cout << "augmentation: " << aug << std::endl;
  }
  else cout << "*****dart prior (Off):\n";
  if(p) cout << "data set: n,p: " << n << ", " << p << std::endl;
  else cout << "data not set\n";
}
