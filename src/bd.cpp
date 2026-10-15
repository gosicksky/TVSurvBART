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

#include "bd.h"

bool bd(tree& x, xinfo& xi, dinfo& di, pinfo& pi, double sigma, 
        std::vector<size_t>& nv, std::vector<double>& pv, bool aug, rn& gen,
        double *fnoh, double *vij, double *ui, tree::tree_cp *leaf)
{
  tree::npv goodbots;  //nodes we could birth at (split on)
  double PBx = getpb(x,xi,pi,goodbots); //prob of a birth at x
  
  if(gen.uniform() < PBx) { //do birth or death
    
    //--------------------------------------------------
    //draw proposal
    tree::tree_p nx; //bottom node
    size_t v,c; //variable and cutpoint
    double pr; //part of metropolis ratio from proposal and prior
    bprop(x,xi,pi,goodbots,PBx,nx,v,c,pr,nv,pv,aug,gen);
    
    //--------------------------------------------------
    //compute sufficient statistics
    size_t nr,nl; //counts in proposed bots
    double syl, syr; //sum of y in proposed bots
    double rhk, rhkl, rhkr; //for survival rhk is for original rhkl and rhkr are for left and right
    double shk, shkl, shkr; //for survival shk is for original shkl and shkr are for left and right
    if(leaf) getsuff(nx,v,c,xi,di,leaf,rhk,rhkl,rhkr,shk,shkl,shkr,vij,ui,fnoh,nl,nr);
    else getsuff(x,nx,v,c,xi,di,rhk,rhkl,rhkr,shk,shkl,shkr,vij,ui,fnoh,nl,nr);
    // Rcpp::Rcout << "rhk:" << rhk << " rhkl:"<<rhkl << " rhkr:" << rhkr << std::endl; 
    // Rcpp::Rcout << "shk:" << shk << " shkl:"<<shkl << " shkr:" << shkr << std::endl; 
    // Rcpp::Rcout << "nr:" << nr << " nl:"<<nl << std::endl; 
    //log linear BART
    double loglikelihood_add = 0;
    
    //print shkl and shkr values for debugging
    //print whether shkl is 0 or shkr is 0
    // Rcpp::Rcout << "shkl: " << shkl << " shkr: " << shkr << std::endl;
    // Rcpp::Rcout << "shkl == 0: " << (shkl == 0.0) << " shkr == 0: " << (shkr == 0.0) << std::endl;
    
    loglikelihood_add = logsumexp_two(logZ(-pi.c + rhkl, 2.0 * pi.d, 2.0 * shkl), logZ(pi.c + rhkl, 0, 2.0 * (pi.d + shkl))) - log(2.0) - logZ(pi.c,0,2.0*pi.d) +  
      logsumexp_two(logZ(-pi.c + rhkr, 2.0 * pi.d, 2.0 * shkr), logZ(pi.c + rhkr, 0, 2.0 * (pi.d + shkr))) - log(2.0) - logZ(pi.c,0,2.0*pi.d);
    
    
    // if (std::isnan(rhk) || std::isnan(shk) || std::isinf(rhk) || std::isinf(shk) || -pi.c + rhk <= 0. || shk <= 0.) {
    //   //std::cerr << "Invalid value for logres" << std::endl;
    //   loglikelihood_add = -10000.0;  // Sentinel value for invalid inputs
    // }else{
    //   Rcpp::Rcout << "Z, left:" << logsumexp_two(logZ(-pi.c + rhkl, 2.0 * pi.d, 2.0 * shkl), logZ(pi.c + rhkl, 0, 2.0 * (pi.d + shkl))) - log(2.0) - logZ(pi.c,0,2.0*pi.d)  << " right:" <<  logsumexp_two(logZ(-pi.c + rhkr, 2.0 * pi.d, 2.0 * shkr), logZ(pi.c + rhkr, 0, 2.0 * (pi.d + shkr))) - log(2.0) - logZ(pi.c,0,2.0*pi.d) << std::endl;
    //   loglikelihood_add = logsumexp_two(logZ(-pi.c + rhkl, 2.0 * pi.d, 2.0 * shkl), logZ(pi.c + rhkl, 0, 2.0 * (pi.d + shkl))) - log(2.0) - logZ(pi.c,0,2.0*pi.d) +  
    //     logsumexp_two(logZ(-pi.c + rhkr, 2.0 * pi.d, 2.0 * shkr), logZ(pi.c + rhkr, 0, 2.0 * (pi.d + shkr))) - log(2.0) - logZ(pi.c,0,2.0*pi.d);
    // }
    double loglikelihood_original = 0;
    loglikelihood_original = logsumexp_two(logZ(-pi.c + rhk, 2.0 * pi.d, 2.0 * shk), logZ(pi.c + rhk, 0, 2.0 * (pi.d + shk))) - log(2.0) - logZ(pi.c,0,2.0*pi.d);
    
    //printf("loglikelihood_add: %f\n", loglikelihood_add);
    //printf("loglikelihood_original: %f\n", loglikelihood_original);
    
    //--------------------------------------------------
    //compute alpha
    double alpha=0.0, lalpha=0.0;
    if((nl>=5) && (nr>=5)) { //cludge?
      // lhl = lh(nl,syl,sigma,pi.tau);
      // lhr = lh(nr,syr,sigma,pi.tau);
      // lht = lh(nl+nr,syl+syr,sigma,pi.tau);
      // 
      // alpha=1.0;
      // lalpha = log(pr) + (lhl+lhr-lht) + log(sigma);
      // lalpha = std::min(0.0,lalpha);
      
      //log linear BART
      alpha = 1.0;
      lalpha = log(pr) + loglikelihood_add - loglikelihood_original;
      lalpha = std::min(0.0,lalpha);
    }
    
    //--------------------------------------------------
    //try metrop
    double mul,mur; //means for new bottom nodes, left and right
    double uu = gen.uniform();
    bool dostep = (alpha > 0) && (log(uu) < lalpha);
    //Rcpp::Rcout << "bd:" << dostep << std::endl; 
    if(dostep) {
      mul = drawnodemu(pi.c,pi.d,rhkl,shkl,gen);
      mur = drawnodemu(pi.c,pi.d,rhkr,shkr,gen);
      x.birthp(nx,v,c,mul,mur);
      if(leaf) { //rows of nx move to its new children (same rule as tree::bn)
        const double cut = xi[v][c];
        for(size_t i=0;i<di.n;i++)
          if(leaf[i]==nx) leaf[i] = ((di.x + i*di.p)[v] < cut) ? nx->getl() : nx->getr();
      }
      nv[v]++;
      return true;
    } else {
      return false;
    }
  } else {
    //--------------------------------------------------
    //draw proposal
    double pr;  //part of metropolis ratio from proposal and prior
    tree::tree_p nx; //nog node to death at
    dprop(x,xi,pi,goodbots,PBx,nx,pr,gen);
    
    //--------------------------------------------------
    //compute sufficient statistics
    size_t nr,nl; //counts at bots of nx
    double syl, syr; //sum at bots of nx
    double rhk; //for survival
    double shk; //for survival
    double rhkl, rhkr; //for survival
    double shkl, shkr; //for survival
    if(leaf) getsuff(nx->getl(), nx->getr(), di, leaf, rhk, rhkl, rhkr, shk, shkl, shkr, vij, ui, fnoh, nl, nr);
    else getsuff(x, nx->getl(), nx->getr(), xi, di, rhk, rhkl, rhkr, shk, shkl, shkr, vij, ui, fnoh, nl, nr);
    
    //log linear bart
    double loglikelihood_sub = logsumexp_two(logZ(-pi.c + rhk, 2 * pi.d, 2 * shk), logZ(pi.c + rhk, 0, 2 * (pi.d + shk))) - log(2.0) - logZ(pi.c,0,2.0*pi.d);
    double loglikelihood_original = logsumexp_two(logZ(-pi.c + rhkl, 2.0 * pi.d, 2.0 * shkl), logZ(pi.c + rhkl, 0, 2.0 * (pi.d + shkl))) - log(2.0) - logZ(pi.c,0,2.0*pi.d) +
      logsumexp_two(logZ(-pi.c + rhkr, 2.0 * pi.d, 2.0 * shkr), logZ(pi.c + rhkr, 0, 2.0 * (pi.d + shkr))) - log(2.0) - logZ(pi.c,0,2.0*pi.d);
    
    // if (std::isnan(rhk) || std::isnan(shk) || std::isinf(rhk) || std::isinf(shk) || -pi.c + rhk <= 0. || shk <= 0.) {
    //   //std::cerr << "Invalid value for logres" << std::endl;
    //   loglikelihood_original = -10000.0;  // Sentinel value for invalid inputs
    // }else{
    //   double loglikelihood_original = logsumexp_two(logZ(-pi.c + rhkl, 2.0 * pi.d, 2.0 * shkl), logZ(pi.c + rhkl, 0, 2.0 * (pi.d + shkl))) - log(2.0) - logZ(pi.c,0,2.0*pi.d) +
    //     logsumexp_two(logZ(-pi.c + rhkr, 2.0 * pi.d, 2.0 * shkr), logZ(pi.c + rhkr, 0, 2.0 * (pi.d + shkr))) - log(2.0) - logZ(pi.c,0,2.0*pi.d);
    // }
    
    //printf("loglikelihood_sub: %f\n", loglikelihood_sub);
    //printf("loglikelihood_original: %f\n", loglikelihood_original);
    
    //log linear BART
    double lalpha = 0.0;
    lalpha = log(pr) +  loglikelihood_sub - loglikelihood_original;
    lalpha = std::min(0.0,lalpha);
    
    //--------------------------------------------------
    //try metrop
    double mu;
    double mu_test;
    if(log(gen.uniform()) < lalpha) {
      //Rcpp::Rcout << "bd:" << true << std::endl; 
      mu = drawnodemu(pi.c,pi.d,rhk,shk,gen);
      nv[nx->getv()]--;
      if(leaf) { //rows of the two children fall back to nx (before deathp frees them)
        tree::tree_cp l = nx->getl(), r = nx->getr();
        for(size_t i=0;i<di.n;i++)
          if(leaf[i]==l || leaf[i]==r) leaf[i] = nx;
      }
      x.deathp(nx,mu);
      return true;
    } else {
      //Rcpp::Rcout << "bd:" << false << std::endl; 
      return false;
    }
  }
}
