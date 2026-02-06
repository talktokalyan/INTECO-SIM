import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(page_title="Mundell-Fleming Simulation", layout="wide")

st.title(" 🌍 Mundell-Fleming Model Simulator")
st.markdown("""
    Use the sidebar to adjust fiscal and monetary policy. 
    The model solves for **Output (Y)**, **Interest Rate (i)**, and **Exchange Rate (e)**.
""")

# --- 2. SIDEBAR INPUTS ---
st.sidebar.header("1. Policy Variables")
G = st.sidebar.slider("Government Spending (G)", 0, 500, 250)
M = st.sidebar.slider("Money Supply (M)", 100, 1000, 500)
T = st.sidebar.slider("Taxes (T)", 0, 500, 200)

st.sidebar.header("2. Market Parameters")
regime = st.sidebar.selectbox("Exchange Rate Regime", ["Floating", "Fixed"])
mobility = st.sidebar.select_slider("Capital Mobility", options=["Low", "High", "Perfect"], value="High")

# Map qualitative mobility to quantitative sigma value
sigma_map = {"Low": 10, "High": 2000, "Perfect": 1000000}
sigma = sigma_map[mobility]

# ... (Previous sidebar code for inputs G, M, T, regime, mobility) ...

st.sidebar.markdown("---") # Adds a visual separator
st.sidebar.header("3. Fixed Model Parameters")

# Display parameters in a clean, mathematical format
st.sidebar.latex(r"""
\begin{aligned}
\small
& \text{Consumption (c)}_1 = 0.75 \\
& \text{Inv. Sensitivity (b)} = 800 \\
& \text{Import Propensity (m)} = 0.1 \\
& \text{NX Sensitivity (x)} = 400 \\
& \text{Money Demand (k)} = 0.5 \\
& \text{Money Sensitivity (h)} = 1200
\end{aligned}
""")

st.sidebar.markdown("---")
with st.sidebar.expander("See Model Equations"):
    st.latex(r"""
    \begin{aligned}
    & \textbf{Unknowns: } [Y, i, e] \\
    \\
    & \textbf{1. IS (Goods Market):} \\
    & Y = C + I + G + NX \\
    & \Rightarrow (1-c_1+m)Y + b i - x e \\
    & \quad = C_0 - c_1T + I_0 + G + NX_0 \\
    \\
    & \textbf{2. LM (Money Market):} \\
    & \frac{M}{P} = kY - h i \\
    & \Rightarrow kY - h i = \frac{M}{P} \\
    \\
    & \textbf{3. BP (Balance of Payments):} \\
    & NX + CF = 0 \\
    & \Rightarrow NX_0 -mY + \sigma (i-i^*) + x e = 0  
    \end{aligned}
    """)

# --- 3. MODEL SOLVER ---
def solve_mundell_fleming(G, T, M, regime, sigma):
    # Parameters
    c1 = 0.75   # Marginal propensity to consume
    b = 80     # Sensitivity of investment to interest
    m = 0.1     # Marginal propensity to import
    x = 40     # Sensitivity of NX to exchange rate
    k = 0.5     # Income sensitivity of money demand
    h = 120    # Interest sensitivity of money demand
    
    # Exogenous Constants
    C0 = 200; I0 = 100; NX0 = 100; i_star = 0.05; P = 1; e_target = 1.0

    # Matrix Construction: A * x = B
    if regime == "Floating":
        # Unknowns: [Y, i, e]
        # 1. IS: Y = C + I + G + NX  => (1-c1+m)Y + bi - xe = C0 - c1T + I0 + G + NX0
        # 2. LM: M/P = kY - hi       => kY - hi = M/P
        # 3. BP: NX + CF = 0         => -mY + sigma*i + xe = sigma*i_star - NX0
        
        A = np.array([
            [1 - c1 + m,  b,  x],
            [k,          -h,   0],
            [-m,      sigma,   -x]
        ])
        B = np.array([
            C0 - c1*T + I0 + G + NX0,
            M / P,
            sigma * i_star - NX0
        ])
        
        Y, i, e = np.linalg.solve(A, B)
        return Y, i, e, M
        
    else: # Fixed Exchange Rate
        # Unknowns: [Y, i, M_endogenous] (e is fixed)
        # 1. IS: (1-c1+m)Y + bi = C0 - c1T + I0 + G + NX0 + x*e_target
        # 2. LM: kY - hi - (1/P)M = 0
        # 3. BP: -mY + sigma*i = sigma*i_star - NX0 - x*e_target
        
        A = np.array([
            [1 - c1 + m,  b,   0],
            [k,          -h, -1/P],
            [-m,      sigma,   0]
        ])
        B = np.array([
            C0 - c1*T + I0 + G + NX0 + x*e_target,
            0,
            sigma * i_star - NX0 - x*e_target
        ])
        
        Y, i, M_endo = np.linalg.solve(A, B)
        return Y, i, e_target, M_endo

# Solve the model
Y_eq, i_eq, e_eq, M_actual = solve_mundell_fleming(G, T, M, regime, sigma)

# --- 4. VISUALIZATION FUNCTIONS ---
def get_curves(Y_range, params, equilibrium):
    Y = Y_range
    Y_val, i_val, e_val, M_val = equilibrium
    
    # Unpack parameters
    c1=0.75; b=800; m=0.1; x=400; k=0.5; h=1200; 
    C0=200; I0=100; NX0=100; i_star=0.05; P=1
    
    # 1. LM Curve: i = (kY - M/P) / h
    i_LM = (k * Y - M_val/P) / h
    
    # 2. IS Curve (drawn at equilibrium exchange rate e_val):
    # (1-c1+m)Y + bi - x*e_val = Intercept
    # bi = Intercept + x*e_val - (1-c1+m)Y
    IS_intercept = C0 - c1*T + I0 + G + NX0
    i_IS = (IS_intercept + x*e_val - (1-c1+m)*Y) / b
    
    # 3. BP Curve (drawn at equilibrium exchange rate e_val):
    # -mY + sigma*i + x*e_val = sigma*i_star - NX0
    # sigma*i = sigma*i_star - NX0 - x*e_val + mY
    if sigma > 10000: # Perfect mobility approximate
        i_BP = np.full_like(Y, i_star)
    else:
        i_BP = (sigma*i_star - NX0 - x*e_val + m*Y) / sigma
        
    return i_IS, i_LM, i_BP

# --- 5. PLOTTING ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Equilibrium Results")
    st.metric("Output (Y)", f"{Y_eq:.1f}")
    st.metric("Interest Rate (i)", f"{i_eq:.2%}")
    st.metric("Exchange Rate (e)", f"{e_eq:.2f}")
    st.caption(f"Regime: {regime} | Mobility: {mobility}")
    if regime == "Fixed":
        st.info(f"Central Bank Adjusted M to: {M_actual:.1f}")

with col2:
    st.subheader("IS-LM-BP Diagram")
    
    # Generate Plot Data
    Y_range = np.linspace(Y_eq - 500, Y_eq + 500, 100)
    i_IS, i_LM, i_BP = get_curves(Y_range, None, (Y_eq, i_eq, e_eq, M_actual))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot curves
    ax.plot(Y_range, i_IS, label="IS Curve (Goods)", color="blue", linewidth=2)
    ax.plot(Y_range, i_LM, label="LM Curve (Money)", color="red", linewidth=2)
    ax.plot(Y_range, i_BP, label="BP Curve (BoP)", color="green", linestyle="--", linewidth=2)
    
    # Plot Equilibrium Point
    ax.scatter([Y_eq], [i_eq], color="black", s=100, zorder=5)
    ax.text(Y_eq, i_eq + 0.005, "  Equilibrium", fontsize=12, fontweight='bold')
    
    # Formatting
    ax.set_xlabel("Output (Y)")
    ax.set_ylabel("Interest Rate (i)")
    ax.set_title(f"Macroeconomic Equilibrium (e = {e_eq:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
