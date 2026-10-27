// =============================================================================
// Regressionstest fuer die Vorzeichen-/Orientierungsfixes in
//   src/Mesh.cpp                (getNedelecCurl, generateCoil)
//   include/NedelecHexahedron.cuh (evaluate_nedelec_basis, evaluate_nedelec_curl)
//
// Laeuft ohne CUDA und ohne GPU:
//   g++ -O2 -std=c++17 -Wall -Wextra -o test_nedelec tests/test_nedelec_orientation.cpp
//   ./test_nedelec        # Exit-Code 0 = alle Tests bestanden
//
// Geprueft wird:
//   A) getNedelecCurl(e,...)        == curl(getNedelecShape(e,...))     numerisch
//   B) evaluate_nedelec_curl(...)   == curl(evaluate_nedelec_basis(...)) numerisch
//   C) CPU- und GPU-Basis sind kollinear mit konstantem Verhaeltnis 0.5
//      (deckt das frueher wechselnde Vorzeichen der Kanten 2,3,6,7 ab)
//   D) generateCoil erzeugt ausschliesslich Elemente mit detJ > 0
//
// ACHTUNG / WARTUNGSHINWEIS: Diese Datei enthaelt *Kopien* der getesteten
// Funktionen, weil NedelecHexahedron.cuh cuda_runtime.h benoetigt und Mesh.h
// transitiv cusparse/cublas/cudss einbindet. Bei Aenderungen an den
// Originalfunktionen muessen die Kopien hier mit angepasst werden, sonst
// testet die Datei stillschweigend den falschen Code.
// =============================================================================

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdlib>
#define PATCHED_CPU 1
#define PATCHED_GPU 1
struct Point3D { double x,y,z; };
struct HexElement { int nodes[8]; };
struct d3 { double x,y,z; };
static d3 mk(double a,double b,double c){return d3{a,b,c};}

// ===== gepatchte CPU-Funktionen (aus Mesh.cpp nach Fix) =====
void getNedelecShape(int e,double xi,double eta,double zeta,double v[3]){
  v[0]=v[1]=v[2]=0;
  switch(e){
  case 0: v[0]= 0.25*(1-eta)*(1-zeta); break;
  case 1: v[1]= 0.25*(1+xi)*(1-zeta);  break;
  case 2: v[0]=-0.25*(1+eta)*(1-zeta); break;
  case 3: v[1]=-0.25*(1-xi)*(1-zeta);  break;
  case 4: v[0]= 0.25*(1-eta)*(1+zeta); break;
  case 5: v[1]= 0.25*(1+xi)*(1+zeta);  break;
  case 6: v[0]=-0.25*(1+eta)*(1+zeta); break;
  case 7: v[1]=-0.25*(1-xi)*(1+zeta);  break;
  case 8: v[2]= 0.25*(1-xi)*(1-eta);   break;
  case 9: v[2]= 0.25*(1+xi)*(1-eta);   break;
  case 10:v[2]= 0.25*(1+xi)*(1+eta);   break;
  case 11:v[2]= 0.25*(1-xi)*(1+eta);   break;
  }
}
void getNedelecCurl(int e,double xi,double eta,double zeta,double c[3]){
  c[0]=c[1]=c[2]=0;
  switch(e){
  case 0: c[1]=-0.25*(1-eta); c[2]=0.25*(1-zeta); break;
  case 1: c[0]= 0.25*(1+xi);  c[2]=0.25*(1-zeta); break;
  case 2: c[1]= 0.25*(1+eta); c[2]=0.25*(1-zeta); break;
  case 3: c[0]=-0.25*(1-xi);  c[2]=0.25*(1-zeta); break;
  case 4: c[1]= 0.25*(1-eta); c[2]=0.25*(1+zeta); break;
  case 5: c[0]=-0.25*(1+xi);  c[2]=0.25*(1+zeta); break;
  case 6: c[1]=-0.25*(1+eta); c[2]=0.25*(1+zeta); break;
  case 7: c[0]= 0.25*(1-xi);  c[2]=0.25*(1+zeta); break;
  case 8: c[0]=-0.25*(1-xi);  c[1]= 0.25*(1-eta); break;
  case 9: c[0]=-0.25*(1+xi);  c[1]=-0.25*(1-eta); break;
  case 10:c[0]= 0.25*(1+xi);  c[1]=-0.25*(1+eta); break;
  case 11:c[0]= 0.25*(1-xi);  c[1]= 0.25*(1+eta); break;
  }
}
// ===== gepatchte GPU-Funktionen (aus NedelecHexahedron.cuh nach Fix) =====
void gpu_basis(double xi,double eta,double zeta,d3 W[12]){
  W[0]=mk( 0.125*(1-eta)*(1-zeta),0,0); W[2]=mk(-0.125*(1+eta)*(1-zeta),0,0);
  W[4]=mk( 0.125*(1-eta)*(1+zeta),0,0); W[6]=mk(-0.125*(1+eta)*(1+zeta),0,0);
  W[1]=mk(0, 0.125*(1+xi)*(1-zeta),0);  W[3]=mk(0,-0.125*(1-xi)*(1-zeta),0);
  W[5]=mk(0, 0.125*(1+xi)*(1+zeta),0);  W[7]=mk(0,-0.125*(1-xi)*(1+zeta),0);
  W[8] =mk(0,0,0.125*(1-xi)*(1-eta));   W[9] =mk(0,0,0.125*(1+xi)*(1-eta));
  W[10]=mk(0,0,0.125*(1+xi)*(1+eta));   W[11]=mk(0,0,0.125*(1-xi)*(1+eta));
}
void gpu_curl(double xi,double eta,double zeta,d3 c[12]){
  c[0]=mk(0,-0.125*(1-eta), 0.125*(1-zeta)); c[2]=mk(0, 0.125*(1+eta), 0.125*(1-zeta));
  c[4]=mk(0, 0.125*(1-eta), 0.125*(1+zeta)); c[6]=mk(0,-0.125*(1+eta), 0.125*(1+zeta));
  c[1]=mk( 0.125*(1+xi),0, 0.125*(1-zeta));  c[3]=mk(-0.125*(1-xi),0, 0.125*(1-zeta));
  c[5]=mk(-0.125*(1+xi),0, 0.125*(1+zeta));  c[7]=mk( 0.125*(1-xi),0, 0.125*(1+zeta));
  c[8] =mk(-0.125*(1-xi), 0.125*(1-eta),0);  c[9] =mk(-0.125*(1+xi),-0.125*(1-eta),0);
  c[10]=mk( 0.125*(1+xi),-0.125*(1+eta),0);  c[11]=mk( 0.125*(1-xi), 0.125*(1+eta),0);
}
void gpu_shape_wrap(int e,double xi,double eta,double zeta,double v[3]){
  d3 W[12]; gpu_basis(xi,eta,zeta,W); v[0]=W[e].x;v[1]=W[e].y;v[2]=W[e].z;
}
template<class F> void numcurl(F f,int e,double xi,double eta,double zeta,double o[3]){
  const double h=1e-6; double p[3],m[3],dA[3][3];
  for(int j=0;j<3;++j){ double c[3]={xi,eta,zeta}; c[j]+=h; f(e,c[0],c[1],c[2],p);
    c[0]=xi;c[1]=eta;c[2]=zeta;c[j]-=h; f(e,c[0],c[1],c[2],m);
    for(int i=0;i<3;++i) dA[i][j]=(p[i]-m[i])/(2*h); }
  o[0]=dA[2][1]-dA[1][2]; o[1]=dA[0][2]-dA[2][0]; o[2]=dA[1][0]-dA[0][1];
}
// ===== Jacobi / Volumen (aus Mesh.cpp) =====
void calcJacobian(const Point3D n[8],double xi,double eta,double zeta,double J[3][3],double& detJ){
  double sX[8]={-1,1,1,-1,-1,1,1,-1}, sY[8]={-1,-1,1,1,-1,-1,1,1}, sZ[8]={-1,-1,-1,-1,1,1,1,1};
  for(int r=0;r<3;++r)for(int c=0;c<3;++c) J[r][c]=0;
  for(int i=0;i<8;++i){
    double a=0.125*sX[i]*(1+sY[i]*eta)*(1+sZ[i]*zeta);
    double b=0.125*(1+sX[i]*xi)*sY[i]*(1+sZ[i]*zeta);
    double c=0.125*(1+sX[i]*xi)*(1+sY[i]*eta)*sZ[i];
    J[0][0]+=a*n[i].x; J[0][1]+=a*n[i].y; J[0][2]+=a*n[i].z;
    J[1][0]+=b*n[i].x; J[1][1]+=b*n[i].y; J[1][2]+=b*n[i].z;
    J[2][0]+=c*n[i].x; J[2][1]+=c*n[i].y; J[2][2]+=c*n[i].z;
  }
  detJ = J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])
       - J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])
       + J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);
}
// ===== gepatchte generateCoil =====
void generateCoil(double rIn,double w,double h,double z0,int nR,int nZ,int nP,
                  std::vector<Point3D>& nodes, std::vector<HexElement>& els){
  const double PI=3.14159265358979323846;
  for(int k=0;k<=nP;++k){ double phi=2*PI*k/nP;
    for(int j=0;j<=nZ;++j){ double z=z0+h*j/nZ;
      for(int i=0;i<=nR;++i){ double r=rIn+w*i/nR; nodes.push_back({r*std::cos(phi),r*std::sin(phi),z}); }}}
  int sl=(nZ+1)*(nR+1);
  for(int k=0;k<nP;++k)for(int j=0;j<nZ;++j)for(int i=0;i<nR;++i){
    int b=k*sl+j*(nR+1)+i, b2=(k+1)*sl+j*(nR+1)+i; HexElement el;
    el.nodes[0]=b;  el.nodes[1]=b+(nR+1);  el.nodes[2]=b+(nR+1)+1;  el.nodes[3]=b+1;
    el.nodes[4]=b2; el.nodes[5]=b2+(nR+1); el.nodes[6]=b2+(nR+1)+1; el.nodes[7]=b2+1;
    els.push_back(el);
  }
}
int main(){
  int fails=0;
  double pts[3][3]={{0.13,-0.27,0.41},{0,0,0},{-0.7,0.55,-0.31}};
  printf("--- Test A: CPU getNedelecCurl == curl(getNedelecShape) ---\n");
  double worst=0;
  for(int t=0;t<3;++t)for(int e=0;e<12;++e){
    double a[3],n[3]; getNedelecCurl(e,pts[t][0],pts[t][1],pts[t][2],a);
    numcurl(getNedelecShape,e,pts[t][0],pts[t][1],pts[t][2],n);
    for(int i=0;i<3;++i) worst=std::fmax(worst,std::fabs(a[i]-n[i]));
  }
  printf("  max. Abweichung ueber 3 Punkte x 12 Kanten: %.3e  -> %s\n", worst, worst<1e-8?"OK":"FEHLER");
  if(worst>=1e-8) fails++;

  printf("--- Test B: GPU evaluate_nedelec_curl == curl(evaluate_nedelec_basis) ---\n");
  worst=0;
  for(int t=0;t<3;++t){ d3 gc[12]; gpu_curl(pts[t][0],pts[t][1],pts[t][2],gc);
    for(int e=0;e<12;++e){ double a[3]={gc[e].x,gc[e].y,gc[e].z},n[3];
      numcurl(gpu_shape_wrap,e,pts[t][0],pts[t][1],pts[t][2],n);
      for(int i=0;i<3;++i) worst=std::fmax(worst,std::fabs(a[i]-n[i])); }}
  printf("  max. Abweichung ueber 3 Punkte x 12 Kanten: %.3e  -> %s\n", worst, worst<1e-8?"OK":"FEHLER");
  if(worst>=1e-8) fails++;

  printf("--- Test C: CPU- und GPU-Basis kollinear, Verhaeltnis konstant 0.5 ---\n");
  double rmin=1e30,rmax=-1e30; bool coll=true;
  for(int t=0;t<3;++t){ d3 W[12]; gpu_basis(pts[t][0],pts[t][1],pts[t][2],W);
    for(int e=0;e<12;++e){ double c[3]; getNedelecShape(e,pts[t][0],pts[t][1],pts[t][2],c);
      double g[3]={W[e].x,W[e].y,W[e].z};
      for(int i=0;i<3;++i){ if(std::fabs(c[i])>1e-12){ double r=g[i]/c[i]; rmin=std::fmin(rmin,r); rmax=std::fmax(rmax,r);} 
                           else if(std::fabs(g[i])>1e-12) coll=false; }}}
  printf("  Verhaeltnis GPU/CPU in [%.6f, %.6f], kollinear=%s -> %s\n", rmin,rmax,coll?"ja":"nein",
         (coll && rmax-rmin<1e-12 && std::fabs(rmin-0.5)<1e-12)?"OK":"FEHLER");
  if(!(coll && rmax-rmin<1e-12 && std::fabs(rmin-0.5)<1e-12)) fails++;

  printf("--- Test D: generateCoil, alle detJ > 0 ---\n");
  std::vector<Point3D> nd; std::vector<HexElement> el;
  generateCoil(27,28,52,0,4,10,40,nd,el);
  int neg=0; double vol=0;
  for(auto&e:el){ Point3D n[8]; for(int i=0;i<8;++i)n[i]=nd[e.nodes[i]];
    double J[3][3],d; calcJacobian(n,0,0,0,J,d); if(d<=0)neg++; vol+=d*8.0; }
  const double PI=3.14159265358979323846;
  double ref=PI*((27+28)*(27+28)-27.0*27)*52;
  printf("  Knoten=%zu Elemente=%zu\n",nd.size(),el.size());
  printf("  Elemente mit detJ<=0: %d von %zu\n",neg,el.size());
  printf("  Summe Volumen=%.2f  Referenz(Hohlzylinder)=%.2f  Verhaeltnis=%.6f\n",vol,ref,vol/ref);
  if(neg!=0) fails++;

  printf("\n%s (%d fehlgeschlagen)\n", fails==0?"ALLE TESTS BESTANDEN":"TESTS FEHLGESCHLAGEN", fails);
  return fails;
}
