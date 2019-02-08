import starrotate as sr
import rotation_tools as rt

# Load the data
KID = "4760478"
# raw_time, raw_flux, raw_flux_err = rt.download_lightcurves(KID)
time, flux, flux_err = rt.load_kepler_data(
    "/Users/ruthangus/.kplr/data/lightcurves/{}".format(str(int(KID))
                                                        .zfill(9)))

# Measure rotation period
star = sr.RotationModel(time, flux, flux_err, KID)
t0, dur, porb = 636.1, 1, 287.4
gp_period, ls_period, acf_period = star.measure_rotation_period(t0, dur, porb)
print("gp_period = ", gp_period, "LS period = ", ls_period, "acf period = ",
      acf_period)
