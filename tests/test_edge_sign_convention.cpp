// =============================================================================
// Regressionstest fuer die Vorzeichen-Konvention der Elementmatrizen
// (Commit "doppelte Vorzeichenkorrektur in assemble_local_matrices_kernel")
//
//   g++ -O2 -std=c++17 -Wall -Wextra -o test_signs tests/test_edge_sign_convention.cpp
//   ./test_signs
//
// Konvention der Pipeline: lokale Elementmatrizen werden UNORIENTIERT
// aufgebaut, die Kantenorientierung wird ausschliesslich beim Gather/Scatter
// angewendet (scatter_to_csr_kernel: sign_r*sign_c).
//
// Der Test baut fuer ein stark verzerrtes Hexaeder die lokale C-Matrix in drei
// Varianten und vergleicht das Ergebnis nach dem Scatter gegen die Referenz,
// die direkt mit global orientierten Basisfunktionen s_i*W_i rechnet:
//   mode 0  ohne die Zeile "real_W[i].x *= edge_signs[i]"   -> erwartet exakt
//   mode 1  mit dieser Zeile (Originalzustand)              -> erwartet falsch
//   mode 2  Referenz
//
// Die fehlerhafte Matrix bleibt symmetrisch, eine Symmetriepruefung findet den
// Fehler also nicht - daher dieser Test.
//
// WARTUNGSHINWEIS: enthaelt eine Kopie der Basis aus NedelecHexahedron.cuh,
// weil diese cuda_runtime.h benoetigt. Bei Aenderungen mitpflegen.
// =============================================================================

#include <cstdio>
#include <cmath>
struct d3{double x,y,z;};
static d3 mk(double a,double b,double c){return d3{a,b,c};}
// gepatchte GPU-Basis (Branch fix/nedelec-orientation)
void basis(double xi,double eta,double z,d3 W[12]){
  W[0]=mk( 0.125*(1-eta)*(1-z),0,0); W[2]=mk(-0.125*(1+eta)*(1-z),0,0);
  W[4]=mk( 0.125*(1-eta)*(1+z),0,0); W[6]=mk(-0.125*(1+eta)*(1+z),0,0);
  W[1]=mk(0, 0.125*(1+xi)*(1-z),0);  W[3]=mk(0,-0.125*(1-xi)*(1-z),0);
  W[5]=mk(0, 0.125*(1+xi)*(1+z),0);  W[7]=mk(0,-0.125*(1-xi)*(1+z),0);
  W[8]=mk(0,0,0.125*(1-xi)*(1-eta)); W[9]=mk(0,0,0.125*(1+xi)*(1-eta));
  W[10]=mk(0,0,0.125*(1+xi)*(1+eta));W[11]=mk(0,0,0.125*(1-xi)*(1+eta));
}
void jac(const d3 n[8],double xi,double eta,double z,double J[3][3],double invJ[3][3],double&det){
  double sX[8]={-1,1,1,-1,-1,1,1,-1},sY[8]={-1,-1,1,1,-1,-1,1,1},sZ[8]={-1,-1,-1,-1,1,1,1,1};
  for(int r=0;r<3;++r)for(int c=0;c<3;++c)J[r][c]=0;
  for(int i=0;i<8;++i){
    double a=0.125*sX[i]*(1+sY[i]*eta)*(1+sZ[i]*z);
    double b=0.125*(1+sX[i]*xi)*sY[i]*(1+sZ[i]*z);
    double c=0.125*(1+sX[i]*xi)*(1+sY[i]*eta)*sZ[i];
    J[0][0]+=a*n[i].x;J[0][1]+=a*n[i].y;J[0][2]+=a*n[i].z;
    J[1][0]+=b*n[i].x;J[1][1]+=b*n[i].y;J[1][2]+=b*n[i].z;
    J[2][0]+=c*n[i].x;J[2][1]+=c*n[i].y;J[2][2]+=c*n[i].z;
  }
  det=J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])-J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])
     +J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);
  double id=1.0/det;
  invJ[0][0]=(J[1][1]*J[2][2]-J[1][2]*J[2][1])*id; invJ[0][1]=(J[0][2]*J[2][1]-J[0][1]*J[2][2])*id;
  invJ[0][2]=(J[0][1]*J[1][2]-J[0][2]*J[1][1])*id; invJ[1][0]=(J[1][2]*J[2][0]-J[1][0]*J[2][2])*id;
  invJ[1][1]=(J[0][0]*J[2][2]-J[0][2]*J[2][0])*id; invJ[1][2]=(J[0][2]*J[1][0]-J[0][0]*J[1][2])*id;
  invJ[2][0]=(J[1][0]*J[2][1]-J[1][1]*J[2][0])*id; invJ[2][1]=(J[0][1]*J[2][0]-J[0][0]*J[2][1])*id;
  invJ[2][2]=(J[0][0]*J[1][1]-J[0][1]*J[1][0])*id;
}
// mode 0 = ohne Zeile 64 (Fix), 1 = mit Zeile 64 (Original), 2 = Referenz (global orientierte Basis)
void buildC(const d3 n[8],const int s[12],double kappa,int mode,double C[12][12]){
  for(int i=0;i<12;++i)for(int j=0;j<12;++j)C[i][j]=0;
  const double g=0.5773502691896257;
  double pts[8][3]={{-g,-g,-g},{g,-g,-g},{-g,g,-g},{g,g,-g},{-g,-g,g},{g,-g,g},{-g,g,g},{g,g,g}};
  for(int q=0;q<8;++q){
    double xi=pts[q][0],eta=pts[q][1],ze=pts[q][2];
    double J[3][3],iJ[3][3],det; jac(n,xi,eta,ze,J,iJ,det);
    d3 W[12],rW[12]; basis(xi,eta,ze,W);
    for(int i=0;i<12;++i){
      rW[i].x=iJ[0][0]*W[i].x+iJ[1][0]*W[i].y+iJ[2][0]*W[i].z;
      rW[i].y=iJ[0][1]*W[i].x+iJ[1][1]*W[i].y+iJ[2][1]*W[i].z;
      rW[i].z=iJ[0][2]*W[i].x+iJ[1][2]*W[i].y+iJ[2][2]*W[i].z;
      if(mode==1) rW[i].x*=s[i];                                  // Original: nur .x
      if(mode==2){ rW[i].x*=s[i]; rW[i].y*=s[i]; rW[i].z*=s[i]; }  // Referenz: global orientiert
    }
    for(int i=0;i<12;++i)for(int j=0;j<12;++j)
      C[i][j]+=kappa*(rW[i].x*rW[j].x+rW[i].y*rW[j].y+rW[i].z*rW[j].z)*det;
  }
}
int main(){
  // stark verzerrtes Hexaeder, damit invJ nicht diagonal ist und x/y/z mischen
  d3 n[8]={{0,0,0},{1.3,0.1,0.05},{1.1,1.2,-0.1},{0.05,0.9,0.1},
           {0.1,-0.05,1.1},{1.2,0.2,1.3},{1.4,1.1,1.2},{-0.1,1.0,0.95}};
  // realistisches Vorzeichenmuster: einige Kanten global umorientiert
  int s[12]={1,-1,1,1,-1,1,-1,1,1,1,-1,-1};
  double kappa=3.4e7;
  double Cfix[12][12],Cbug[12][12],Cref[12][12];
  buildC(n,s,kappa,0,Cfix); buildC(n,s,kappa,1,Cbug); buildC(n,s,kappa,2,Cref);
  // Scatter wendet s_r*s_c an
  double mf=0,mb=0,scale=0;
  for(int i=0;i<12;++i)for(int j=0;j<12;++j){
    double fix=Cfix[i][j]*s[i]*s[j], bug=Cbug[i][j]*s[i]*s[j], ref=Cref[i][j];
    scale=std::fmax(scale,std::fabs(ref));
    mf=std::fmax(mf,std::fabs(fix-ref)); mb=std::fmax(mb,std::fabs(bug-ref));
  }
  printf("Typischer Matrixwert (max |C_ref|) : %.6e\n",scale);
  printf("Ohne Zeile 64 (Fix)  : max|C_assembled - C_ref| = %.6e  (rel %.3e)\n",mf,mf/scale);
  printf("Mit Zeile 64 (Original): max|C_assembled - C_ref| = %.6e  (rel %.3e)\n",mb,mb/scale);
  // wie viele Eintraege sind betroffen?
  int bad=0; for(int i=0;i<12;++i)for(int j=0;j<12;++j){
    double bug=Cbug[i][j]*s[i]*s[j];
    if(std::fabs(bug-Cref[i][j])>1e-9*scale) bad++;
  }
  printf("Betroffene Eintraege der 12x12 C-Matrix: %d von 144\n",bad);
  // Symmetrie bleibt erhalten -> Fehler ist durch Symmetriepruefung NICHT auffindbar
  double asym=0; for(int i=0;i<12;++i)for(int j=0;j<12;++j)
    asym=std::fmax(asym,std::fabs(Cbug[i][j]-Cbug[j][i]));
  printf("Asymmetrie der fehlerhaften C-Matrix: %.3e  (bleibt symmetrisch!)\n",asym);
  return 0;
}
