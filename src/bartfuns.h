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

#ifndef GUARD_bartfuns_h
#define GUARD_bartfuns_h

#include "tree.h"
#include "treefuns.h"
#include "info.h"
#include "distribution.h"
#include <algorithm>

//--------------------------------------------------
//make xinfo = cutpoints
void makexinfo(size_t p, size_t n, double *x, xinfo& xi, size_t nc);
void makexinfo(size_t p, size_t n, double *x, xinfo& xi, int* nc);
//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
double getpb(tree& t, xinfo& xi, pinfo& pi, tree::npv& goodbots);
//--------------------------------------------------
//compute n and \sum y_i for left and right give bot and v,c
void getsuff(tree& x, tree::tree_p nx, size_t v, size_t c, xinfo& xi, dinfo& di, size_t& nl, double& syl, size_t& nr, double& syr);
void getsuff(tree& x, tree::tree_p nx, size_t v, size_t c, xinfo& xi, dinfo& di, double& rhk, double& rhkl, double& rhkr, double& shk, double& shkl, double& shkr, double *vij, double *ui, double *fnoh, size_t& nl, size_t& nr);
//--------------------------------------------------
//lh, replacement for lil that only depends on sum y.
double lh(size_t n, double sy, double sigma, double tau);
//--------------------------------------------------
//get prob a node grows, 0 if no good vars, else a/(1+d)^b
double pgrow(tree::tree_p n, xinfo& xi, pinfo& pi);
//--------------------------------------------------
//compute n and \sum y_i for left and right bots
void getsuff(tree& x, tree::tree_p l, tree::tree_p r, xinfo& xi, dinfo& di, size_t& nl, double& syl, size_t& nr, double& syr);
void getsuff(tree& x, tree::tree_p l, tree::tree_p r, xinfo& xi, dinfo& di, double& rhk, double& rhkl, double& rhkr, double& shk, double& shkl, double& shkr, double *vij, double *ui, double *fnoh, size_t& nl, size_t& nr);
//--------------------------------------------------
//get sufficients stats for all bottom nodes, this way just loop through all the data once.
void allsuff(tree& x, xinfo& xi, dinfo& di, tree::npv& bnv, std::vector<double>& rhk, std::vector<double>& shk, double *fnoh, double *vij, double *ui);
void allsuff(tree& x, xinfo& xi, dinfo& di, tree::npv& bnv, std::vector<size_t>& nv, std::vector<double>& syv);
//--------------------------------------------------
//survival sufficient statistics from a per-row bottom-node cache (leaf[i] = bottom
//node of row i in the tree): the same sums, in the same row order, as the versions
//above, without descending the tree for every row.
//birth at bottom node nx on (v,c); death of the children l, r; all bottom nodes
void getsuff(tree::tree_cp nx, size_t v, size_t c, xinfo& xi, dinfo& di, const tree::tree_cp* leaf, double& rhk, double& rhkl, double& rhkr, double& shk, double& shkl, double& shkr, double *vij, double *ui, double *fnoh, size_t& nl, size_t& nr);
void getsuff(tree::tree_cp l, tree::tree_cp r, dinfo& di, const tree::tree_cp* leaf, double& rhk, double& rhkl, double& rhkr, double& shk, double& shkl, double& shkr, double *vij, double *ui, double *fnoh, size_t& nl, size_t& nr);
void allsuff(tree& x, dinfo& di, const tree::tree_cp* leaf, tree::npv& bnv, std::vector<double>& rhk, std::vector<double>& shk, double *fnoh, double *vij, double *ui);
//--------------------------------------------------
// draw all the bottom node mu's
void drmu(tree& t, xinfo& xi, dinfo& di, pinfo& pi, double sigma, rn& gen, double *fnoh, double *vij, double *ui);
void drmu(tree& t, xinfo& xi, dinfo& di, pinfo& pi, double sigma, rn& gen);
//survival version using the tree's per-row bottom-node cache
void drmu(tree& t, xinfo& xi, dinfo& di, pinfo& pi, double sigma, rn& gen, double *fnoh, double *vij, double *ui, const tree::tree_cp* leaf);
//--------------------------------------------------
//birth proposal
void bprop(tree& x, xinfo& xi, pinfo& pi, tree::npv& goodbots, double& PBx, tree::tree_p& nx, size_t& v, size_t& c, double& pr, std::vector<size_t>& nv, std::vector<double>& pv, bool aug, rn& gen);
//--------------------------------------------------
// death proposal
void dprop(tree& x, xinfo& xi, pinfo& pi, tree::npv& goodbots, double& PBx, tree::tree_p& nx, double& pr, rn& gen);
//--------------------------------------------------
//draw one mu from post 
double drawnodemu(size_t n, double sy, double tau, double sigma, rn& gen);
double drawnodemu(double c, double d, double rhk, double shk, rn& gen);
//--------------------------------------------------
//draw variable splitting probabilities from Dirichlet (Linero, 2018)
void draw_s(std::vector<size_t>& nv, std::vector<double>& lpv, double& theta, rn& gen);
void draw_s_grp(std::vector<size_t>& nv, std::vector<double>& lpv, double& theta, rn& gen, double* grp, double rho=0.);
//--------------------------------------------------
//draw Dirichlet sparsity parameter from posterior using grid
void draw_theta0(bool const_theta, double& theta, std::vector<double>& lpv,
		 double a, double b, double rho, rn& gen);
#endif
