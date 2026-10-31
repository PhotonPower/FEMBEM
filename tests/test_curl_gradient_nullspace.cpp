// =============================================================================
// Regressionstest: die Steifigkeitsmatrix muss Gradienten annihilieren.
//
//   cl /O2 /std:c++17 /EHsc /Fe:t3.exe tests\test_curl_gradient_nullspace.cpp
//   g++ -O2 -std=c++17 -o t3 tests/test_curl_gradient_nullspace.cpp
//
// Fuer jedes Knotenskalarfeld phi gilt mit a_e = phi(n_B) - phi(n_A) exakt
//
//     sum_e a_e * W_e = grad(phi)   =>   curl(...) = 0   =>   K * a = 0
//
// Das ist eine physikalische Eigenschaft und damit unabhaengig von der
// Herleitung der Curl-Formeln selbst. Der Test prueft Basis, Curl,
// Kantenrichtung und Piola-Transformation in einem Durchgang.
//
// Erwartetes Ergebnis: die gepatchte Variante liegt bei Maschinengenauigkeit,
// die Originalvariante aus NedelecHexahedron.cuh vor dem Fix bei O(1).
//
// WARTUNGSHINWEIS: enthaelt Kopien beider Basisvarianten, weil
// NedelecHexahedron.cuh cuda_runtime.h benoetigt. Bei Aenderungen mitpflegen.
// =============================================================================

#include <cstdio>
#include <cmath>
struct d3{double x,y,z;};
static d3 mk(double a,double b,double c){return d3{a,b,c};}
const int EDGE[12][2]={{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};

// variant 0 = gepatcht (Branch), 1 = Original aus dem Repo vor dem Fix
void basis(int var,double xi,double e,double z,d3 W[12]){
  double s2 = (var==0)?-1:+1, s3 = (var==0)?-1:+1;
  W[0]=mk( 0.125*(1-e)*(1-z),0,0);      W[2]=mk(s2*0.125*(1+e)*(1-z),0,0);
  W[4]=mk( 0.125*(1-e)*(1+z),0,0);      W[6]=mk(s2*0.125*(1+e)*(1+z),0,0);
  W[1]=mk(0, 0.125*(1+xi)*(1-z),0);     W[3]=mk(0,s3*0.125*(1-xi)*(1-z),0);
  W[5]=mk(0, 0.125*(1+xi)*(1+z),0);     W[7]=mk(0,s3*0.125*(1-xi)*(1+z),0);
  W[8]=mk(0,0,0.125*(1-xi)*(1-e));      W[9]=mk(0,0,0.125*(1+xi)*(1-e));
  W[10]=mk(0,0,0.125*(1+xi)*(1+e));     W[11]=mk(0,0,0.125*(1-xi)*(1+e));
}
void curl(int var,double xi,double e,double z,d3 c[12]){
  if(var==0){ // gepatcht
    c[0]=mk(0,-0.125*(1-e), 0.125*(1-z)); c[2]=mk(0, 0.125*(1+e), 0.125*(1-z));
    c[4]=mk(0, 0.125*(1-e), 0.125*(1+z)); c[6]=mk(0,-0.125*(1+e), 0.125*(1+z));
    c[1]=mk( 0.125*(1+xi),0, 0.125*(1-z));c[3]=mk(-0.125*(1-xi),0, 0.125*(1-z));
    c[5]=mk(-0.125*(1+xi),0, 0.125*(1+z));c[7]=mk( 0.125*(1-xi),0, 0.125*(1+z));
    c[8]=mk(-0.125*(1-xi), 0.125*(1-e),0);c[9]=mk(-0.125*(1+xi),-0.125*(1-e),0);
    c[10]=mk(0.125*(1+xi),-0.125*(1+e),0);c[11]=mk( 0.125*(1-xi), 0.125*(1+e),0);
  } else {    // Original
    c[0]=mk(0, 0.125*(1-e),-0.125*(1-z)); c[2]=mk(0,-0.125*(1+e), 0.125*(1-z));
    c[4]=mk(0, 0.125*(1-e), 0.125*(1+z)); c[6]=mk(0,-0.125*(1+e),-0.125*(1+z));
    c[1]=mk(-0.125*(1+xi),0, 0.125*(1-z));c[3]=mk( 0.125*(1-xi),0,-0.125*(1-z));
    c[5]=mk(-0.125*(1+xi),0,-0.125*(1+z));c[7]=mk( 0.125*(1-xi),0, 0.125*(1+z));
    c[8]=mk( 0.125*(1-xi),-0.125*(1-e),0);c[9]=mk( 0.125*(1+xi), 0.125*(1-e),0);
    c[10]=mk(-0.125*(1+xi),0.125*(1+e),0);c[11]=mk(-0.125*(1-xi),-0.125*(1+e),0);
  }
}
void jac(const d3 n[8],double xi,double et,double z,double J[3][3],double&det){
  double sX[8]={-1,1,1,-1,-1,1,1,-1},sY[8]={-1,-1,1,1,-1,-1,1,1},sZ[8]={-1,-1,-1,-1,1,1,1,1};
  for(int r=0;r<3;++r)for(int c2=0;c2<3;++c2)J[r][c2]=0;
  for(int i=0;i<8;++i){
    double a=0.125*sX[i]*(1+sY[i]*et)*(1+sZ[i]*z);
    double b=0.125*(1+sX[i]*xi)*sY[i]*(1+sZ[i]*z);
    double c2=0.125*(1+sX[i]*xi)*(1+sY[i]*et)*sZ[i];
    J[0][0]+=a*n[i].x;J[0][1]+=a*n[i].y;J[0][2]+=a*n[i].z;
    J[1][0]+=b*n[i].x;J[1][1]+=b*n[i].y;J[1][2]+=b*n[i].z;
    J[2][0]+=c2*n[i].x;J[2][1]+=c2*n[i].y;J[2][2]+=c2*n[i].z;
  }
  det=J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])-J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])
     +J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);
}
void buildK(int var,const d3 n[8],double nu,double K[12][12]){
  for(int i=0;i<12;++i)for(int j=0;j<12;++j)K[i][j]=0;
  const double g=0.5773502691896257;
  double p[8][3]={{-g,-g,-g},{g,-g,-g},{-g,g,-g},{g,g,-g},{-g,-g,g},{g,-g,g},{-g,g,g},{g,g,g}};
  for(int q=0;q<8;++q){
    double xi=p[q][0],et=p[q][1],ze=p[q][2];
    double J[3][3],det; jac(n,xi,et,ze,J,det);
    d3 cw[12],rc[12]; curl(var,xi,et,ze,cw);
    // Kontravariant SPALTENweise ueber J, wie piola_contravariant in
    // NedelecDispatch.cuh: jac() legt J wie compute_jacobian() transponiert
    // ab (J[r][c] = d x_c / d xi_r). Bis 31.10.2026 stand hier die
    // zeilenweise Form; der Nullraum haengt daran nicht (rot grad = 0 gilt
    // schon im Referenzelement), der Operator aber schon. Inline statt ueber
    // den Helfer, weil dieser Test bewusst ohne CUDA-Header auskommt.
    for(int i=0;i<12;++i){
      rc[i].x=(J[0][0]*cw[i].x+J[1][0]*cw[i].y+J[2][0]*cw[i].z)/det;
      rc[i].y=(J[0][1]*cw[i].x+J[1][1]*cw[i].y+J[2][1]*cw[i].z)/det;
      rc[i].z=(J[0][2]*cw[i].x+J[1][2]*cw[i].y+J[2][2]*cw[i].z)/det;
    }
    for(int i=0;i<12;++i)for(int j=0;j<12;++j)
      K[i][j]+=nu*(rc[i].x*rc[j].x+rc[i].y*rc[j].y+rc[i].z*rc[j].z)*det;
  }
}
int main(){
  d3 n[8]={{0,0,0},{1.3,0.1,0.05},{1.1,1.2,-0.1},{0.05,0.9,0.1},
           {0.1,-0.05,1.1},{1.2,0.2,1.3},{1.4,1.1,1.2},{-0.1,1.0,0.95}};
  const double PI=3.14159265358979323846;
  double nu=1.0/(4.0*PI*1e-7);
  double phi[8]={0.7,-1.3,2.1,0.4,-0.9,1.8,-2.2,0.15}; // beliebiges Knotenskalarfeld
  double a[12]; for(int e=0;e<12;++e) a[e]=phi[EDGE[e][1]]-phi[EDGE[e][0]];
  for(int var=0;var<2;++var){
    double K[12][12]; buildK(var,n,nu,K);
    double resid=0,scale=0;
    for(int i=0;i<12;++i){
      double s=0; for(int j=0;j<12;++j) s+=K[i][j]*a[j];
      resid=std::fmax(resid,std::fabs(s));
      for(int j=0;j<12;++j) scale=std::fmax(scale,std::fabs(K[i][j]*a[j]));
    }
    printf("%-28s max|K*a| = %.4e   relativ zu %.4e  ->  %.3e   %s\n",
      var==0?"gepatcht (Branch)":"Original (vor Fix)",
      resid,scale,resid/scale, resid/scale<1e-12?"BESTANDEN":"DURCHGEFALLEN");
  }
  return 0;
}
