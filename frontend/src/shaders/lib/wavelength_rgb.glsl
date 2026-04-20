// Approximate CIE wavelength → sRGB. Classic piecewise polynomial from Dan
// Bruton's "approximate RGB values for visible wavelengths" — fast, no LUT,
// good enough for iridescent rainbows.
vec3 wavelength_to_rgb(float nm) {
  float r = 0.0, g = 0.0, b = 0.0;
  if (nm < 380.0 || nm > 780.0) return vec3(0.0);
  if (nm < 440.0)       { r = -(nm - 440.0) / 60.0; g = 0.0;                    b = 1.0; }
  else if (nm < 490.0)  { r = 0.0;                  g = (nm - 440.0) / 50.0;    b = 1.0; }
  else if (nm < 510.0)  { r = 0.0;                  g = 1.0;                    b = -(nm - 510.0) / 20.0; }
  else if (nm < 580.0)  { r = (nm - 510.0) / 70.0;  g = 1.0;                    b = 0.0; }
  else if (nm < 645.0)  { r = 1.0;                  g = -(nm - 645.0) / 65.0;   b = 0.0; }
  else                  { r = 1.0;                  g = 0.0;                    b = 0.0; }
  // Intensity falloff at the spectrum edges so the rainbow fades naturally.
  float a = 1.0;
  if (nm < 420.0)      a = 0.3 + 0.7 * (nm - 380.0) / 40.0;
  else if (nm > 700.0) a = 0.3 + 0.7 * (780.0 - nm) / 80.0;
  return a * vec3(r, g, b);
}
