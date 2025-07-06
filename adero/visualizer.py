import math
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from adero.cst import cst_model

def _get_cross_section(wing, y):
    span     = math.sqrt(wing["AR"] * wing["area"])
    frac     = 0 if span == 0 else y / span
    root_c   = wing["chord"]
    tip_c    = root_c * wing["taper"]
    chord    = root_c + (tip_c - root_c) * frac
    sweep_le = math.tan(math.radians(wing["sweep"]))

    θ     = np.linspace(0, math.pi, 50)
    xnorm = 0.5 * (1 - np.cos(θ))

    # surfaces at root & tip, then linear interp
    ur = cst_model(xnorm, *wing["CST_coeff"]["root"]["upper"]) * root_c
    ut = cst_model(xnorm, *wing["CST_coeff"]["tip"]["upper"])  * tip_c
    lr = cst_model(xnorm, *wing["CST_coeff"]["root"]["lower"]) * root_c
    lt = cst_model(xnorm, *wing["CST_coeff"]["tip"]["lower"])  * tip_c

    upper = ur * (1 - frac) + ut * frac
    lower = lr * (1 - frac) + lt * frac
    x_le   = y * sweep_le
    x      = xnorm * chord + x_le

    return x, upper, lower

def plot_rib(wing, idx, num, thickness):
    # total and half-span
    span     = math.sqrt(wing["AR"] * wing["area"])

    # margin from root & tip = 5% of half-span
    margin   = 0.05 * span

    # evenly spaced rib centers from margin to (semi-margin)
    if num > 1:
        y_positions = np.linspace(margin, span - margin, num)
    else:
        y_positions = np.array([span / 2.0])

    # this rib’s centre
    y_center = y_positions[idx]

    # get the exact cross-section at that centre
    x, upper, lower = _get_cross_section(wing, y_center)

    # build the outline (upper forward, lower back)
    outline_x = np.concatenate([x, x[::-1]])
    outline_z = np.concatenate([upper, lower[::-1]])

    # clamp front/back faces to lie within [0, semi]
    y_front = max(0.0, y_center - thickness/2)
    y_back  = min(span, y_center + thickness/2)

    # assemble vertices
    verts_front = [(outline_x[i], y_front, outline_z[i]) for i in range(len(outline_x))]
    verts_back  = [(outline_x[i], y_back,  outline_z[i]) for i in range(len(outline_x))]
    cent_f = (np.mean(outline_x), y_front, np.mean(outline_z))
    cent_b = (np.mean(outline_x), y_back,  np.mean(outline_z))
    vertices = [cent_f] + verts_front + [cent_b] + verts_back

    # triangulate front cap
    n = len(outline_x)
    faces = []
    for i in range(1, n):
        faces.append((0, i, i+1))
    faces.append((0, n, 1))

    # triangulate back cap
    offset = n + 1
    for i in range(offset+1, offset+n):
        faces.append((offset, i, i+1))
    faces.append((offset, offset+n, offset+1))

    # connect sides
    for i in range(1, n):
        faces.append((i, i+1, offset+i))
        faces.append((offset+i, i+1, offset+i+1))
    faces.append((n, 1, offset+n))
    faces.append((offset+n, 1, offset+1))

    i_idx, j_idx, k_idx = zip(*faces)
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]

    return go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=i_idx, j=j_idx, k=k_idx,
        color="orange", opacity=1,
        name=f"Rib {idx+1}"
    )

def plot_spar(wing_params, idx, thickness):
    if idx not in [0, 1]:
        raise ValueError("spar_index must be 0 or 1")

    span      = math.sqrt(wing_params["AR"] * wing_params["area"])
    root_c    = wing_params["chord"]
    tip_c     = root_c * wing_params["taper"]
    sweep_le  = math.tan(math.radians(wing_params["sweep"]))
    frac_list = [0.25, 0.75]
    frac      = frac_list[idx]

    # leading-edge x offset
    x_le_root = 0
    x_le_tip  = span * sweep_le

    x_root = x_le_root + root_c * frac
    x_tip  = x_le_tip    + tip_c  * frac

    z_ur = cst_model(np.array([frac]), *wing_params["CST_coeff"]["root"]["upper"])[0] * root_c
    z_lr = cst_model(np.array([frac]), *wing_params["CST_coeff"]["root"]["lower"])[0] * root_c
    z_ut = cst_model(np.array([frac]), *wing_params["CST_coeff"]["tip"]["upper"])[0]  * tip_c
    z_lt = cst_model(np.array([frac]), *wing_params["CST_coeff"]["tip"]["lower"])[0]  * tip_c

    half = thickness / 2
    # 8 vertices: root lower-left, root lower-right, root upper-right, root upper-left,
    # tip lower-left, tip lower-right, tip upper-right, tip upper-left
    verts = [
        (x_root-half, 0,        z_lr),
        (x_root+half, 0,        z_lr),
        (x_root+half, 0,        z_ur),
        (x_root-half, 0,        z_ur),
        (x_tip -half, span,     z_lt),
        (x_tip +half, span,     z_lt),
        (x_tip +half, span,     z_ut),
        (x_tip -half, span,     z_ut),
    ]
    faces = [
        (0,1,5),(0,5,4),  # bottom
        (3,2,6),(3,6,7),  # top
        (0,3,7),(0,7,4),  # left side
        (1,2,6),(1,6,5),  # right side
        (0,1,2),(0,2,3),  # front face
        (4,5,6),(4,6,7),  # back face
    ]
    i,j,k = zip(*faces)
    xv = [v[0] for v in verts]
    yv = [v[1] for v in verts]
    zv = [v[2] for v in verts]

    return go.Mesh3d(x=xv, y=yv, z=zv, i=i, j=j, k=k,
                     color="green", opacity=1, name=f"Spar {idx+1}")

def generate_wing_figure(wing_params, upper_opacity=1, lower_opacity=1, show_internal=True):
    """
    Generate the outer 3D wing surfaces using CST-based airfoil definitions.
    upper_opacity: 1 to show the upper skin, 0 to hide.
    lower_opacity: 1 to show the lower skin, 0 to hide.
    If show_internal is True, add the internal ribs and spars.
    """
    sweep_rad = math.radians(wing_params["sweep"])
    span      = math.sqrt(wing_params["AR"] * wing_params["area"])
    tip_chord = wing_params["chord"] * wing_params["taper"]
    x_le_tip  = span * math.tan(sweep_rad)

    # Build the two surfaces (root → tip) exactly as before
    num_x, num_y = 50, 200
    theta        = np.linspace(0, np.pi, num_x)
    X_vals       = 0.5 * (1 - np.cos(theta))
    X_root       = wing_params["chord"] * X_vals
    X_tip        = x_le_tip + tip_chord * X_vals

    Z_root_upper = cst_model(X_vals, *wing_params["CST_coeff"]["root"]["upper"]) * wing_params["chord"]
    Z_root_lower = cst_model(X_vals, *wing_params["CST_coeff"]["root"]["lower"]) * wing_params["chord"]
    Z_tip_upper  = cst_model(X_vals, *wing_params["CST_coeff"]["tip"]["upper"])  * tip_chord
    Z_tip_lower  = cst_model(X_vals, *wing_params["CST_coeff"]["tip"]["lower"])  * tip_chord

    # Interpolate across span
    X_full = np.array([np.linspace(X_root[i], X_tip[i], num_y) for i in range(num_x)]).T
    Y_full = np.tile(np.linspace(0, span, num_y), (num_x, 1)).T
    Z_full_upper = np.array([np.linspace(Z_root_upper[i], Z_tip_upper[i], num_y) for i in range(num_x)]).T
    Z_full_lower = np.array([np.linspace(Z_root_lower[i], Z_tip_lower[i], num_y) for i in range(num_x)]).T

    fig = go.Figure()

    # Add upper/lower skins based on the opacity flags
    if upper_opacity:
        fig.add_trace(go.Surface(
            x=X_full, y=Y_full, z=Z_full_upper,
            colorscale="Blues", showscale=False,
            opacity=upper_opacity, name="Upper Surface"
        ))
    if lower_opacity:
        fig.add_trace(go.Surface(
            x=X_full, y=Y_full, z=Z_full_lower,
            colorscale="Reds", showscale=False,
            opacity=lower_opacity, name="Lower Surface"
        ))

    # Internal structure
    if show_internal:
        rib_t   = wing_params["internal"]["rib_thickness"]
        num_r   = wing_params["internal"]["num_ribs"]
        for i in range(num_r):
            fig.add_trace(plot_rib(wing_params, idx=i, num=num_r, thickness=rib_t))
        spar_t = wing_params["internal"]["spar_thickness"]
        for si in [0, 1]:
            fig.add_trace(plot_spar(wing_params, idx=si, thickness=spar_t))

    fig.update_layout(
        scene=dict(
            xaxis_title="Chordwise (m)",
            yaxis_title="Spanwise (m)",
            zaxis_title="Thickness (m)",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title="CST-Based Wing Surface"
    )
    return fig


def generate_2d_airfoil_plot(wing_params):
    """
    Flexible 2D airfoil plot function.
    Supports:
      - full wings: CST_coeff["root"]["upper"], ["tip"]["upper"], etc.
      - single airfoils: CST_coeff["upper"], ["lower"]
    """

    fig = None

    if "root" in wing_params["CST_coeff"]:  # Full wing with root/tip
        root_chord = wing_params.get("chord", 1.0)
        tip_chord = root_chord * wing_params.get("taper", 1.0)

        # Root
        x = np.linspace(0, 1, 200)
        root_upper = cst_model(x, *wing_params["CST_coeff"]["root"]["upper"]) * root_chord
        root_lower = cst_model(x, *wing_params["CST_coeff"]["root"]["lower"]) * root_chord

        # Tip
        tip_upper = cst_model(x, *wing_params["CST_coeff"]["tip"]["upper"]) * tip_chord
        tip_lower = cst_model(x, *wing_params["CST_coeff"]["tip"]["lower"]) * tip_chord

        fig = sp.make_subplots(rows=2, cols=1, subplot_titles=("Scaled CST Airfoils", "Normalized CST Airfoils"))

        fig.add_trace(go.Scatter(x=x * root_chord, y=root_upper, mode='lines', name="Root Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x * root_chord, y=root_lower, mode='lines', name="Root Lower"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x * tip_chord, y=tip_upper, mode='lines', name="Tip Upper"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x * tip_chord, y=tip_lower, mode='lines', name="Tip Lower"), row=1, col=1)

        # Normalized
        fig.add_trace(go.Scatter(x=x, y=root_upper / root_chord, mode='lines', showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=root_lower / root_chord, mode='lines', showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=tip_upper / tip_chord, mode='lines', showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=x, y=tip_lower / tip_chord, mode='lines', showlegend=False), row=2, col=1)

        fig.update_layout(height=800, title="CST Root/Tip Airfoil Comparison", template='plotly_white')

    else:  # Single airfoil input
        x = np.linspace(0, 1, 200)
        upper = cst_model(x, *wing_params["CST_coeff"]["upper"])
        lower = cst_model(x, *wing_params["CST_coeff"]["lower"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=upper, mode='lines', name="Upper"))
        fig.add_trace(go.Scatter(x=x, y=lower, mode='lines', name="Lower"))
        fig.update_layout(title="CST Airfoil", template='plotly_white', yaxis_scaleanchor='x')

    return fig