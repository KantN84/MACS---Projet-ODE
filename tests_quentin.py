import numpy as np
# from numba import njit
import matplotlib.pyplot as plt

# Initial conditions
Atmosphere_Initial = 750
CarbonateRock_Initial = 100000000
DeepOcean_Initial = 38000
FossilFuel_Initial = 7500
Plant_Initial = 560
Soil_Initial = 1500
SurfaceOcean_Initial = 890
VegLandArea_percent_Initial = 100

x0 = np.array([Atmosphere_Initial,
                CarbonateRock_Initial,
                DeepOcean_Initial,
                FossilFuel_Initial,
                Plant_Initial,
                Soil_Initial,
                SurfaceOcean_Initial,
                VegLandArea_percent_Initial
                ], dtype=float)

# Constants
Alk = 2.222446077610055
Kao = .278
SurfOcVol = .0362
Deforestation = 0

# Helper functions
# @njit
def AtmCO2(Atmosphere):
    return Atmosphere * (280/Atmosphere_Initial)
# @njit
def GlobalTemp(AtmCO2):
    return 15 + ((AtmCO2-280) * .01)
# @njit
def CO2Effect(AtmCO2):
    return 1.5 * ((AtmCO2) - 40) / ((AtmCO2) + 80)
# @njit
def WaterTemp(GlobalTemp):
    return 273+GlobalTemp
# @njit
def TempEffect(GlobalTemp):
    return ((60 - GlobalTemp) * (GlobalTemp + 15)) / (((60 + 15) / 2) ** (2))/.96
# @njit
def SurfCConc(SurfaceOcean):
    return (SurfaceOcean/12000)/SurfOcVol
# @njit
def Kcarb(WaterTemp):
    return .000575+(.000006*(WaterTemp-278))
# @njit
def KCO2(WaterTemp):
    return .035+(.0019*(WaterTemp-278))
# @njit
def HCO3(Kcarb, SurfCConc):
    return(SurfCConc-(np.sqrt(SurfCConc**2-Alk*(2*SurfCConc-Alk)*(1-4*Kcarb))))/(1-4*Kcarb)
# @njit
def CO3(HCO3):
    return (Alk-HCO3)/2
# @njit
def pCO2Oc(KCO2, HCO3, CO3):
    return 280*KCO2*(HCO3**2/CO3)


# Fossil fuels
FossFuelData = np.array([[1850.0, 0.00], [1875.0, 0.30], [1900.0, 0.60], [1925.0, 1.35], [1950.0, 2.85], [1975.0, 4.95], [2000.0, 7.20], [2025.0, 10.05], [2050.0, 14.85], [2075.0, 20.70], [2100.0, 30.00]])
# CO2 equivalent of 10.05 Gt carbon is 36.88 Gt CO2


# @njit
def FossilFuelsCombustion(t):
    i = 0
    if t >= FossFuelData[-1,0]:
        return FossFuelData[-1,1]
    while i + 1 < len(FossFuelData) and t >= FossFuelData[i,0]:
        i = i + 1
    if i == 0:
        return FossFuelData[0,1]
    else:
        return FossFuelData[i-1,1] + (t - FossFuelData[i-1,0]) / (FossFuelData[i,0] - FossFuelData[i-1,0]) * (FossFuelData[i,1] - FossFuelData[i-1,1])

# @njit
def derivative(x, t):
    Atmosphere = x[0]
    CarbonateRock = x[1]
    DeepOcean = x[2]
    FossilFuelCarbon = x[3]
    Plants = x[4]
    Soils = x[5]
    SurfaceOcean = x[6]
    VegLandArea_percent = x[7]

    PlantResp = (Plants * (55/Plant_Initial)) + Deforestation/2
    Litterfall = (Plants* (55/Plant_Initial))+(Deforestation/2)
    SoilResp = Soils * (55/Soil_Initial)
    Volcanoes = 0.1
    AtmCO2_ = AtmCO2(Atmosphere)
    GlobalTemp_ = GlobalTemp(AtmCO2_)
    WaterTemp_ = WaterTemp(GlobalTemp_)
    Photosynthesis = 110 * CO2Effect(AtmCO2_) * (VegLandArea_percent/100) * TempEffect(GlobalTemp_)
    HCO3_ = HCO3(Kcarb(WaterTemp_), SurfCConc(SurfaceOcean))
    pCO2Oc_ = pCO2Oc(KCO2(WaterTemp_), HCO3_, CO3(HCO3_))
    AtmOcExchange = Kao*(AtmCO2_-pCO2Oc_)
    if x[3] > 0:
        FossilFuelsCombustion_ = FossilFuelsCombustion(t)
    else:
        FossilFuelsCombustion_ = 0
    dAtmosphere_dt = (PlantResp + SoilResp + Volcanoes + FossilFuelsCombustion_
                            - Photosynthesis - AtmOcExchange)

    Sedimentation = DeepOcean * (0.1/DeepOcean_Initial)
    dCarbonateRock_dt = Sedimentation - Volcanoes

    Downwelling = SurfaceOcean*(90.1/SurfaceOcean_Initial)
    Upwelling = DeepOcean * (90/DeepOcean_Initial)
    dDeepOcean_dt= Downwelling - Sedimentation - Upwelling

    dFossilFuelCarbon_dt = - FossilFuelsCombustion_

    dPlants_dt = Photosynthesis - PlantResp - Litterfall

    dSoils_dt = Litterfall - SoilResp

    dSurfaceOcean_dt = Upwelling + AtmOcExchange - Downwelling

    Development = (Deforestation/Plant_Initial * 0.2) * 100
    dVegLandArea_percent_dt = - Development

    derivative = np.array([
        dAtmosphere_dt,
        dCarbonateRock_dt,
        dDeepOcean_dt,
        dFossilFuelCarbon_dt,
        dPlants_dt,
        dSoils_dt,
        dSurfaceOcean_dt,
        dVegLandArea_percent_dt
        ])

    return derivative


def f(t, x):
    return derivative(x, t)


def solve(phi, f, t0, T, x0, h):
    t = np.arange(t0, T+h, h)
    x = np.zeros((len(t), len(x0)))
    x[0] = x0.copy()
    print(t)
    print(x)
    for i in range(len(t)-1):
        x[i+1] = phi(f, t[i], x[i], h) 
    return t, x

def euler(f, t, x, h):
    return x + h * f(t, x)

def runge_kutta_2(f, t, x, h):
    p1 = f(t, x)
    p2 = f(t + h, x + h * p1)
    return x + h * (p1/2 + p2/2)

def runge_kutta_4(f, t, x, h):
    p1 = f(t, x)
    p2 = f(t + h/2, x + h/2 * p1)
    p3 = f(t + h/2, x + h/2 * p2)
    p4 = f(t + h, x + h * p3)
    return x + h * (p1/6 + 2*p2/6 + 2*p3/6 + p4/6)



if __name__ == "__main__":
    t0 = 1850
    T = 2100
    h = 1
    print("Lancement simulation")

    print("Lancement Euler")
    t_euler, x_euler = solve(euler, f, t0, T, x0, h)
    print("Lancement Runge-Kutta 2")
    t_rk2, x_rk2 = solve(runge_kutta_2, f, t0, T, x0, h)
    print("Lancement Runge-Kutta 4")
    t_rk4, x_rk4 = solve(runge_kutta_4, f, t0, T, x0, h)

    print(f"  CO2 final Euler : {x_euler[-1, 0] * 280/750:.1f} ppm")
    print(f"  CO2 final RK2   : {x_rk2[-1, 0] * 280/750:.1f} ppm")
    print(f"  CO2 final RK4   : {x_rk4[-1, 0] * 280/750:.1f} ppm")
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(t_euler, x_euler[:, 0] * 280/750, label="Atmosphere")
    plt.plot(t_euler, x_euler[:, 1], label="Carbonate Rock")
    plt.plot(t_euler, x_euler[:, 2], label="Deep Ocean")
    plt.plot(t_euler, x_euler[:, 3], label="Fossil Fuel Carbon")
    plt.plot(t_euler, x_euler[:, 4], label="Plants")
    plt.plot(t_euler, x_euler[:, 5], label="Soils")
    plt.plot(t_euler, x_euler[:, 6], label="Surface Ocean")
    plt.plot(t_euler, x_euler[:, 7], label="Veg Land Area %")
    plt.xlabel("Year")
    plt.ylabel("Amount")
    plt.title("Carbon Cycle Simulation (Euler)")
    plt.autoscale()
    plt.legend()
    plt.show()



