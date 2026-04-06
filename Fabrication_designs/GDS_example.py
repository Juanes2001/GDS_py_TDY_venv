import numpy as np
import gdsfactory as gf


############################################################
# COSINE S-BEND FUNCTION (adiabatic curvature)
############################################################

def cosine_s_bend(z, D, L):
    return (D / 2) * (1 - np.cos(np.pi * z / L))


############################################################
# ADIABATIC Y-BRANCH WITH ORIGIN AT SPLIT START
############################################################

def adiabatic_y_branch(
    w0=0.5,
    w_out=0.5,
    gap=0.3,
    L_taper=25.0,
    L_branch=40.0,
):

    c = gf.Component("adiabatic_y_branch")

    # Cross-sections
    xs_in = gf.cross_section.strip(width=w0, layer=(1, 0))
    xs_out = gf.cross_section.strip(width=w_out, layer=(1, 0))

    ########################################################
    # 1️⃣ Input straight
    ########################################################
    input_wg = gf.components.straight(length=100, cross_section=xs_in)
    ref_in = c << input_wg

    ########################################################
    # 2️⃣ Taper (single → wide section)
    ########################################################
    taper = gf.components.taper(
        length=L_taper,
        width1=w0,
        width2=2*w_out + gap,
        cross_section=xs_in,
    )
    ref_taper = c << taper
    ref_taper.connect("o1", ref_in.ports["o2"])

    ########################################################
    # 3️⃣ Adiabatic S-bend arms
    ########################################################
    npoints = 300
    z = np.linspace(0, L_branch, npoints)

    D = (gap + w_out)

    y_upper = cosine_s_bend(z, D, L_branch)
    y_lower = -cosine_s_bend(z, D, L_branch)

    path_upper = gf.Path(np.column_stack([z, y_upper]))
    path_lower = gf.Path(np.column_stack([z, y_lower]))

    upper = c << path_upper.extrude(xs_out)
    lower = c << path_lower.extrude(xs_out)

    upper.connect("o1", ref_taper.ports["o2"])
    lower.connect("o1", ref_taper.ports["o2"])

    ########################################################
    # 4️⃣ Output straights
    ########################################################
    out1 = gf.components.straight(length=100, cross_section=xs_out)
    out2 = gf.components.straight(length=100, cross_section=xs_out)

    ref_out1 = c << out1
    ref_out2 = c << out2

    ref_out1.connect("o1", upper.ports["o2"])
    ref_out2.connect("o1", lower.ports["o2"])

    ########################################################
    # 5️⃣ SHIFT ORIGIN TO BRANCHING START
    ########################################################
    # Branching start is the end of taper
    branch_start_x = ref_taper.ports["o2"].center[0]

    # Move entire component so that branch start becomes x = 0
    c.move(origin=(branch_start_x, 0), destination=(0, 0))

    return c


############################################################
# EXPORT
############################################################

if __name__ == "__main__":
    gf.gpdk.PDK.activate()

    y = adiabatic_y_branch(
        w0=0.5,
        w_out=0.5,
        gap=0.3,
        L_taper=25.0,
        L_branch=40.0,
    )

    y.write_gds("horizontal_Y_S_bend_adiabatic.gds")
    y.show()