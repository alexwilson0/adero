import math

g = 9.80665
R = 287.0
layers = [ (11000, -0.0065), (20000, 0.0), (32000, 0.001),
           (47000, 0.0028), (51000, 0.0), (71000, -0.0028), (86000, -0.002) ]

def isa(alt):
    if not 0 <= alt <= 86000:
        raise ValueError("Altitude out of range")
    p0, T0, h0 = 101325, 288.15, 0
    for h1, a in layers:
        if alt <= h1:
            if a:
                T1 = T0 + a * (alt - h0)
                p1 = p0 * (T1 / T0) ** (-g / (a * R))
            else:
                T1 = T0
                p1 = p0 * math.exp(-g * (alt - h0) / (R * T0))
            break
        # Advance to next layer
        T0, h0 = (T0 + a*(h1-h0), h1)
        p0 = p1 if 'p1' in locals() else p0
    rho = p1 / (R * T1)
    # Sutherland's law
    mu0, Tref, S = 1.7894e-5, 288.15, 110.4
    mu = mu0 * (T1 / Tref)**1.5 * ((Tref + S) / (T1 + S))
    return T1, p1, rho, mu