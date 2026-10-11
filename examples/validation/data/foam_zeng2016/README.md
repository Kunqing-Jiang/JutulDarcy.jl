# Digitized reference curves of Zeng et al. (2016), Fig. 5

`reference.csv` holds points read off the three panels of Fig. 5 (gas
saturation, aqueous surfactant concentration in g/L, apparent viscosity in
cP versus x/L at 0.5 PV) for the volumetric partition coefficients
K = 0.05, 1 and 4: Zeng, Ma, Farajzadeh, Puerto, Biswal and Hirasaki (2016),
"Effect of Surfactant Partitioning Between Gaseous Phase and Aqueous Phase
on CO2 Foam Transport for Enhanced Oil Recovery", Transport in Porous Media
114:777–793, DOI 10.1007/s11242-016-0743-6, Case I (CO2 displacing water),
Fig. 5 at 0.50 TPV. The digitized image is the embedded raster of PDF
page 11 (printed page 787). `reference_metadata.json` records the PDF and
image SHA-256, the raster size, the axis pixel coordinates and the sampling
stride.

## Extraction procedure

The publisher PDF is not redistributed. The curves were extracted once with
a Python script (JutulDarcy `e3f18a88`, `examples/foam/zeng2016/extract_reference.py`,
pymupdf, numpy, Pillow); the script was removed when the validation tooling
became Julia-only, so the algorithm is recorded here:

1. Take the largest embedded image of PDF page 11; its RGB raster must be
   912 rows by 1181 columns, otherwise the axis calibration below is invalid.
2. Curve masks by colour, with channel values in 0–255:
   K = 0.05 (blue): `b > 140 && b > r + 75 && b > g + 75`;
   K = 1 (green): `g > 140 && g > r + 75 && g > b + 75`;
   K = 4 (red): `r > 140 && r > g + 75 && r > b + 75`.
3. Panels (image rows of the axis rectangle, top and bottom, and the axis
   maximum): gas_saturation 42–232, 1.0; csw_g_L 350–540, 6.0;
   mu_app_cP 658–848, 30.0. The x axis spans columns 78–1176 for x/L = 0–1.
4. For each panel and colour, blank the legend: rows top to top + 119,
   columns 954 to the right edge.
5. Sample columns 84, 92, …, 1164 (stride 8, 136 columns). In each column
   take the mask pixels in rows top ≤ row < bottom within the five-column
   window col − 2 … col + 2; skip the column when fewer than three pixels
   match; otherwise the curve row is the median of the matching row indices
   (the mean of the two middle values when their number is even).
6. Convert: x/L = (col − 78)/(1176 − 78); value = (bottom − row)/(bottom − top) · axis maximum.

Overlapping curves, the legend area and the clipped part of the K = 0.05
surfactant curve are omitted, never interpolated. One pixel corresponds to
dx/L = 0.000911, dSg = 0.00526, dCsw = 0.0316 g/L, dμ = 0.158 cP; the
practical digitization error is several pixels. The comparison metrics use
only points with 0.05 ≤ x/L ≤ 0.95.

## Contents and checksum

`reference.csv` has the header `quantity,K_volume,x_over_L,value` and 1028
points. `quantity` is `gas_saturation` (–), `csw_g_L` (g/L) or `mu_app_cP`
(cP); `K_volume` is the volumetric partition coefficient; `x_over_L` is
dimensionless. SHA-256:
`0237d978ee02485a95ad6cdf6629bac31d251dba371f685393b60365bdeada7b`
(identical to `foam/zeng2016/data/reference.csv` of the validation
repository, where it is listed in `foam/MANIFEST_frozen.sha256`).

## Use

`examples/validation/10_validation_foam_zeng2016.jl` reads these files
through `pkgdir(JutulDarcy, "examples", "validation", "data", "foam_zeng2016")`.
The same data, together with the validation drivers, historical baselines
and the comparison tooling, is kept in the separate validation repository
`jutul_work/validation/foam/zeng2016/`.
