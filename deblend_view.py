import os
import glob

import flow_vis
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from astropy.io import fits
from PIL import Image


def resolve_n_post(n):

    n = int(n[-1])

    if n==1:
        return "st"
    elif n==2:
        return "nd"
    elif n==3:
        return "rd"
    else:
        return "th"



def sigmoid(x):
    return (1 + np.exp(-x))**-1


def rescale(x):
    return np.interp(x, (x.min(), x.max()), (0, 1))


def get_data()->np.ndarray:
    flx = fits.getdata(os.path.join("./inputs", "1903-flux.fits"))
    com = fits.getdata(os.path.join("output_actual", "com.fits"))
    suppressed = fits.getdata(os.path.join("output_actual", "suppressed.fits"))
    cm = fits.getdata(os.path.join("output_actual", "cm.fits"))
    cv = fits.getdata(os.path.join("output_actual", "cv.fits"))

    n_srcs = len(glob.glob("./output_actual/*[0-9].fits"))
    srcs = [fits.getdata(f"./output_actual/{i}.fits") for i in range(n_srcs)]

    return dict(
        flux = flx,
        com = com,
        suppressed = suppressed,
        cm = cm,
        cv = cv,
        srcs = srcs
    )


def scale_flux(arr):
    m = np.mean(arr)
    s = np.std(arr[np.abs(arr) < m])

    arr = sigmoid(arr + s) #shift to make noise more apparent
    arr = 2*(arr-0.5)
    arr = np.clip(arr,0,1)
    return arr

# https://stackoverflow.com/a/28147716/2691018
def colorize_array(data, vmin=0, vmax=1, cmap=plt.cm.magma):
    normed = plt.Normalize(vmin=vmin, vmax=vmax)
    return cmap(normed(data))


def main():
    data = get_data()

    # ==========================================================================
    # Sidebar 
    # ==========================================================================
    select_flux_scaling = st.sidebar.selectbox(
        "Flux scaling", 
        ["Linear", "Log"]
    )

    select_band = st.sidebar.selectbox(
        "Band",
        [i for i in range(data["flux"].shape[-1])]
    )

    select_n = st.sidebar.select_slider(
        "nth Closest Source",
        [str(i) for i in range(1, data["cm"].shape[-1]+1)]
    )

    select_src = st.sidebar.select_slider(
        "Select Deblended Source",
        [str(i) for i in range(1, len(data["srcs"])+1)]
    )

    select_blend_alpha = st.sidebar.slider(
        "Select Source Mask Alpha",
        min_value=0.0, 
        max_value=1.0, 
        step=0.1,
        value=0.5,
    )
    # ==========================================================================


    st.write("""
    # Morpheus-Deblend View
    """)


    # ==========================================================================
    # Input Flux Display
    # ==========================================================================
    flux = data["flux"][...,select_band]
    if select_flux_scaling=="Linear":
        flux_img = scale_flux(flux)
    elif select_flux_scaling=="Log":
        flux_img = rescale(
            np.log10(
                flux - flux.min() + 1e-3
            )
        )

    st.write("### Input Image")
    st.image(flux_img,use_column_width=True)
    # ==========================================================================
        

    # ==========================================================================
    # Center of Mass
    # ==========================================================================
    com = data["com"]
    suppressed = data["suppressed"]

    st.write("### Identified Sources")
    
    col_com_raw, col_com_suppressed = st.beta_columns(2)
    col_com_raw.write("Center of Mass Raw")
    col_com_raw.image(colorize_array(com[...,0]), use_column_width=True)
    col_com_suppressed.write("Center of Mass Suppressed")
    col_com_suppressed.image(suppressed, use_column_width=True)
    # ==========================================================================


    # ==========================================================================
    # Claim Vectors/Claim Maps
    # ==========================================================================
    n = int(select_n) - 1
    cm = data["cm"][...,select_band, n]
    cv = flow_vis.flow_to_color(
        data["cv"][:, :, n, [1,0]]
    )

    nth_string = select_n + resolve_n_post(select_n)
    col_cv, col_cm = st.beta_columns(2)
    col_cv.write(f"Claim Vector for {nth_string} Source")
    col_cv.image(cv, use_column_width=True)
    col_cm.write(f"Claim Map for {nth_string} Source")
    col_cm.image(colorize_array(cm), use_column_width=True)
    # ==========================================================================


    # ==========================================================================
    # Deblended Sources
    # ==========================================================================
    n_src = int(select_src) - 1
    src = data["srcs"][n_src][:, :, select_band]


    if select_flux_scaling=="Linear":
        src_img = scale_flux(src)
    elif select_flux_scaling=="Log":
        src_img = rescale(
            np.log10(
                src - src.min() + 1e-3
            )
        )
    mask = src != 0
    mask_img = np.zeros(list(mask.shape) + [4], dtype=np.float32)
    
    mask_img[mask, :] = np.array([1,0,0,1.0])
    #mask_img[~mask, :] = np.array([0,0,0,0])

    rgba_flux = (np.dstack(
        [flux_img, flux_img, flux_img, np.ones_like(flux_img)]
        ) * 255).astype(np.uint8)

    img1 = Image.fromarray((mask_img * 255).astype(np.uint8))
    img2 = Image.fromarray(rgba_flux)
    blended = Image.blend(img1, img2, select_blend_alpha)



    st.write("Source {select_src}")
    col_mask, col_src_flux = st.beta_columns(2)
    col_mask.image(np.array(blended), use_column_width=True)
    col_src_flux.image(src_img, use_column_width=True)

    # ==========================================================================



if __name__=="__main__":
    main()