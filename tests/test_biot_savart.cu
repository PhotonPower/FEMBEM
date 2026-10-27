// =============================================================================
// test_biot_savart.cu -- Validierung der analytischen Referenz
// =============================================================================
//
// Die Biot-Savart-Loesung dient als unabhaengige Vergleichsgroesse fuer
// --verify-field. Bevor sie etwas beweisen kann, muss sie selbst stimmen.
// Geprueft wird gegen zwei Grenzfaelle mit geschlossener Loesung und die
// Konvergenz der Quadratur:
//
//   1. Duenne Kreisschleife:  B_z = mu_0 I a^2 / (2 (a^2+d^2)^(3/2))
//   2. Langer duenner Solenoid im Zentrum:  B_z -> mu_0 N I / L
//   3. Konvergenz des zeta-Integrals in der Gauss-Ordnung
//   4. Ausgabe des TEAM-28-Referenzprofils
//
// Braucht weder GPU noch Netz.
// =============================================================================

#include "BiotSavart.cuh"
#include <cstdio>
#include <cmath>
#include <initializer_list>
static constexpr double MU_0 = 4.0e-7*3.14159265358979323846;
int main(){
    int fails=0;
    auto ck=[&](bool ok,const char*w){ printf("  [%s] %s\n",ok?"OK ":"FEHL",w); if(!ok)++fails; };

    // --- 1. Grenzfall duenne Schleife: B_z = mu0 I a^2/(2(a^2+d^2)^{3/2})
    printf("[1] Grenzfall duenne Kreisschleife\n");
    const double a=0.05, I=1.0;
    double worst=0;
    for(double d : {0.0,0.01,0.05,0.2}){
        const double eps=1e-7;                       // Querschnitt eps x eps
        const double J=I/(eps*eps);
        const double num=biot_savart::coil_bz_on_axis(a-eps/2,a+eps/2,-eps/2,eps/2,J,d,400);
        const double ex =MU_0*I*a*a/(2.0*std::pow(a*a+d*d,1.5));
        const double rel=std::fabs(num-ex)/std::fabs(ex);
        worst=std::fmax(worst,rel);
        printf("      d=%.3f m: numerisch %.12e, analytisch %.12e, rel %.2e\n",d,num,ex,rel);
    }
    ck(worst<1e-8,"stimmt mit der Schleifenformel");

    // --- 2. Grenzfall langer Solenoid: B_z im Zentrum -> mu0 * n * I
    printf("\n[2] Grenzfall langer duenner Solenoid, B_z im Zentrum -> mu0*N*I/L\n");
    { const double r=0.01, L=4.0, t=1e-6, NI=1000.0;
      const double J=NI/(t*L);
      const double num=biot_savart::coil_bz_on_axis(r-t/2,r+t/2,-L/2,L/2,J,0.0,2000);
      const double ex =MU_0*NI/L;
      printf("      numerisch %.9e, analytisch %.9e, rel %.2e\n",num,ex,std::fabs(num-ex)/ex);
      ck(std::fabs(num-ex)/ex<1e-4,"Solenoidgrenzwert"); }

    // --- 3. Konvergenz in der Gaussordnung
    printf("\n[3] Konvergenz des zeta-Integrals (TEAM 28, z = 3.8 mm, i = 20 A)\n");
    double prev=0;
    for(int n : {20,50,100,200,400}){
        const double v=biot_savart::coil_bz_on_axis(0.027,0.055,-0.052,0.0,
                        960*20.0/((0.055-0.027)*0.052),0.0038,n);
        printf("      n=%3d: %.15e\n",n,v);
        if(n>20) printf("             Aenderung gegen vorher: %.2e\n",std::fabs(v-prev)/std::fabs(v));
        prev=v;
    }
    ck(true,"Konvergenz siehe oben");

    // --- 4. Das eigentliche Referenzprofil
    printf("\n[4] TEAM-28-Referenzprofil B_z auf der Achse bei i = 20 A (Strommaximum)\n");
    printf("      %-10s %-16s %-16s %-16s\n","z [mm]","B_z gesamt [T]","nur w1","nur w2");
    for(double zmm : {0.0,1.0,3.8,5.3,6.8,11.3,18.2,30.0,50.0}){
        const double z=zmm*1e-3;
        const double b =biot_savart::team28_bz_on_axis(z,20.0);
        const double b1=biot_savart::coil_bz_on_axis(0.027,0.055,-0.052,0.0,
                          960*20.0/((0.055-0.027)*0.052),z);
        const double b2=biot_savart::coil_bz_on_axis(0.080,0.095,-0.052,0.0,
                         -576*20.0/((0.095-0.080)*0.052),z);
        printf("      %-10.2f %-+16.9e %-+16.9e %-+16.9e\n",zmm,b,b1,b2);
    }
    printf("\n%s\n", fails?"FEHLGESCHLAGEN":"REFERENZ VALIDIERT");
    return fails?1:0;
}
