import pandas as pd

import matplotlib.pyplot as plt

from scipy.signal import find_peaks, savgol_filter



hojde_m = float(input("Personens højde i meter (fx 1.75): ").replace(",", "."))



df = pd.read_csv("landmarks.csv")

fps = 1 / df.time_s.diff().median()



# Kalibrering: næse-til-hæl i pixels

hæl_y = (df.l_heel_y + df.r_heel_y) / 2

krop_px_t = hæl_y - df.nose_y

krop_px = krop_px_t.median()

px_per_m = krop_px / (hojde_m * 0.90) # antagelse: næse-til-hæl ≈ 90 % af højden



# Ankelafstand i pixels, glattet

dist_px = (df.l_ankle_x - df.r_ankle_x).abs()

dist_glat = savgol_filter(dist_px, window_length=7, polyorder=2)



# Find skridt

peaks, _ = find_peaks(dist_glat, distance=int(fps * 0.3),

prominence=0.05 * krop_px)



# Resultater

skridt_m = dist_glat[peaks] / px_per_m

tid = df.time_s.iloc[peaks[-1]] - df.time_s.iloc[peaks[0]]

kadence = (len(peaks) - 1) / tid * 60

hastighed = skridt_m.mean() * kadence / 60



print(f"Skridt fundet: {len(peaks)}")

print(f"Skridtlængde (gns.): {skridt_m.mean():.2f} m")

print(f"Skridtlængde pr. skridt: {[round(s, 2) for s in skridt_m]}")

print(f"Kadence: {kadence:.0f} skridt/min")

print(f"Gangshastighed: {hastighed:.2f} m/s")



# Grafer

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))



ax1.plot(df.time_s, dist_px, alpha=0.4, label="rå")

ax1.plot(df.time_s, dist_glat, label="glattet")

ax1.plot(df.time_s.iloc[peaks], dist_glat[peaks], "rx")

ax1.set_ylabel("Ankelafstand (pixels)")

ax1.legend()



ax2.plot(df.time_s, krop_px_t)

ax2.set_ylabel("Næse-til-hæl (pixels)")

ax2.set_xlabel("Tid (s)")



plt.tight_layout()

plt.show()