import pandas as pd
import starrotate as sr
from starrotate.rotation_tools import download_light_curves


def test_transit_mask():

    i = 0

    # Load Stephen Kane's catalogue.
    full_df = pd.read_csv("data/kane_cks_tdmra_dr2.csv")
    df = full_df.drop_duplicates(subset="kepid")

    starname = str(int(df.kepid.values[i])).zfill(9)

    lcpath = "data/lightcurves/{}".format(starname)
    time, flux, flux_err = download_light_curves(df.kepid.values[i], ".",
                                                    lcpath)

    # Measure rotation period¬
    rotate = sr.RotationModel(time, flux, flux_err, starname, plot=True)
    t0, dur, porb = df.koi_time0.values[i], df.koi_duration.values[i], \
        df.koi_period.values[i]
    rotate.process_light_curve(t0, dur, porb)

    # ls_period = rotate.LS_rotation()
    # acf_period = rotate.ACF_rotation()

    # gp_stuff = rotate.GP_rotation()
    # gp_period, errp, errm, Q, Qerrp, Qerrm = gp_stuff


if __name__ == "__main__":
    test_transit_mask()
