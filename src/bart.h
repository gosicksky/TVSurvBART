/*
 *  BART: Bayesian Additive Regression Trees
 *  Copyright (C) 2017-2018 Robert McCulloch, Rodney Sparapani
 *                          and Charles Spanbauer
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

#ifndef GUARD_bart_h
#define GUARD_bart_h

#include <ctime>

#include "info.h"
#include "bartfuns.h"

class bart {
public:
   //------------------------------
   //friends
//    friend bool bd(tree& x, xinfo& xi, dinfo& di, pinfo& pi, double sigma,
// 		  std::vector<size_t>& nv, std::vector<double>& pv, bool aug, rn& gen);
   //------------------------------
   //constructor/destructor
   bart();
   bart(size_t m);
   bart(const bart&);
   ~bart();
   //------------------------------
   //operators
   bart& operator=(const bart&);
   //------------------------------
   //get,set
   size_t getm() {return m;}
   void setm(size_t m);
   
   //void setdata(size_t p, size_t n, double *x, double *y, size_t nc=100);
   void setdata(size_t p, size_t n, size_t nSubject, double *x, double *xObsTime, double *xIndex, double *xEvent, double *y, size_t nc=100);
   //void setdata(size_t p, size_t n, double *x, double *y, int* nc);
   void setdata(size_t p, size_t n, size_t nSubject, double *x, double *xObsTime, double *xIndex, double *xEvent, double *y, int* nc);
   void setprior(double alpha, double beta, double tau, double c, double d)
   {pi.alpha=alpha; pi.mybeta=beta; pi.tau=tau; pi.c=c; pi.d=d;}
   void setpi(pinfo& pi) {this->pi = pi;}
   void setdart(double _a, double _b, int *_grp, bool _aug, bool _dart, 
                double _rho=0., double _theta=0., double _omega=1.) {
     grp= new double[p];
     for(size_t i=0; i<p; ++i) grp[i]=_grp[i];
     if(_rho==0.) for(size_t i=0; i<p; ++i) _rho += 1./grp[i];
     this->setdart(_a, _b, _rho, _aug, _dart, _theta, _omega); 
   }
   
   void setdart(double _a, double _b, double _rho, bool _aug, bool _dart, 
		double _theta=0., double _omega=1.) {
     this->a=_a; this->b=_b; this->rho=_rho; this->aug=_aug; 
     this->dart=_dart; this->omega=_omega; 
     if(_theta==0.){
       this->const_theta=false;
       this->theta=1.;
     }
     else{
       this->const_theta=true;
       this->theta=_theta;
     }
}
   void startdart() {this->dartOn=!(this->dartOn);}
   void settau(double tau) {pi.tau=tau;}
   tree& gettree(size_t i ) { return t[i];}
   
   void settree(std::string itv) {
     std::stringstream ttss(itv);
     size_t _nd, _m, _p;
     //int _nd, _m, _p;
     ttss >> _nd >> _m >> _p;
     cout << _nd << ' ' << _m << ' ' << _p << '\n';
     cout << t.size() << '\n';
     for(size_t j=0; j<m; j++) {
       ttss >> t[j];
       if(j<5) cout << t[j];
     }
   }
   
   xinfo& getxinfo() {return xi;}
   void setxinfo(xinfo& _xi);
   std::vector<size_t>& getnv() {return nv;}
   std::vector<double>& getpv() {return pv;}
   
   void setpv(double *varprob) {
     for(size_t j=0;j<p;j++) pv[j]=varprob[j];
   }
   
   double gettheta() {return theta;}
   //------------------------------
   //public methods
   void birth(size_t i, size_t nid,size_t v, size_t c, double ml, double mr)
         {t[i].birth(nid,v,c,ml,mr);}
   void death(size_t i,size_t nid, double mu)
         {t[i].death(nid,mu);}
   void pr();
   void tonull() {for(size_t i=0;i!=t.size();i++) t[i].tonull();}
   void predict(size_t p, size_t n, double *x, double *fp);
   void draw(double sigma, rn& gen);
//   void draw_s(rn& gen);
   double f(size_t i) {return allfit[i];}
   double getaccept() {return accept;}
protected:
   size_t m;  //number of trees
   std::vector<tree> t; //the trees
   pinfo pi; //prior and mcmc info
   //data
   size_t p,n,nSubject; //x has dim p, n obserations
   double *x,*xObsTime,*xIndex,*xEvent,*y;  //x is column stack, pxn, y is n-vector, xObsTime is n-vector, xIndex is (nsubject+1)-vector, xEvent is (nsubject)-vector
   xinfo xi; //cutpoint info
   //working
   double *allfit; //if the data is set, should be f(x)
   double *r;
   double *ftemp;
   
   double *nodeid; //node id for each observation
   double *fnoh; //f(x) without the current tree
   
   dinfo di;
   bool dart,dartOn,aug,const_theta;
   
   double a,b,rho,theta,omega,accept; // MH acceptance rate from most recent draw
   double *grp;
   
   std::vector<size_t> nv;
   std::vector<double> pv, lpv;
   
   struct nuts_data {
     Eigen::VectorXd xObsTime;
     Eigen::VectorXd xIndex;
     Eigen::VectorXd xEvent;
     Eigen::VectorXd y; // ith y is *(y+i) or y[i]
     int ReSubjectId;
   } nutsData;
};

#endif
