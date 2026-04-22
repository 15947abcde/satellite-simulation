import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import kv, gamma as GammaFunc

# ===============================
# 1. 参数设置
# ===============================
CLWC = np.arange(0, 11, 1)   # ★ 每 1 mg/m^3 一个结果
gamma_th = 0.5   # -3 dB

alpha_fso = 3.2
beta_fso  = 2.1

a1, b1 = 2.5, 1.8
a2, b2 = 3.0, 2.2

gamma_rf_bar = 25
m_rf = 2

# 云衰减参数（增强版）
k_cloud = 0.20   # 云衰减系数
L_cloud = 1      # km

# ===============================
# 2. Gamma–Gamma CDF（云衰减进阈值）
# ===============================
def GG_CDF(gamma_th_eff, gamma_bar, alpha, beta):

    upper = alpha * beta * np.sqrt(gamma_th_eff / gamma_bar)
    eps = 1e-12
    t = np.linspace(eps, upper, 2000)

    y = t**(alpha - 1) * kv(alpha - beta, 2 * np.sqrt(t))
    y = np.nan_to_num(y)

    integral = np.trapz(y, t)
    return integral / (GammaFunc(alpha) * GammaFunc(beta))

# ===============================
# 3. 中断概率计算
# ===============================
Pout_FSO = []
Pout_Hybrid = []
Pout_Proposed = []

for clwc in CLWC:

    # 云衰减（指数模型）
    h_fc = np.exp(-k_cloud * clwc * L_cloud)

    # 等效提高 SNR 阈值
    gamma_th_eff = gamma_th / (h_fc**2)

    # -------- FSO–FSO --------
    gamma_bar_fso = 20
    P_fso = GG_CDF(gamma_th_eff, gamma_bar_fso,
                   alpha_fso, beta_fso)

    # -------- RF --------
    P_rf = (gamma_th / gamma_rf_bar)**m_rf

    # -------- Hybrid --------
    P_hybrid = P_fso * P_rf

    # -------- UAV-FSO --------
    alpha_eq = min(a1, a2)
    beta_eq  = min(b1, b2)
    gamma_bar_uav = 15

    gamma_th_uav = gamma_th / (h_fc**2)
    P_uav = GG_CDF(gamma_th_uav, gamma_bar_uav,
                   alpha_eq, beta_eq)

    # -------- Proposed --------
    P_prop = P_fso * P_uav * P_rf

    Pout_FSO.append(P_fso)
    Pout_Hybrid.append(P_hybrid)
    Pout_Proposed.append(P_prop)

Pout_FSO = np.array(Pout_FSO)
Pout_Hybrid = np.array(Pout_Hybrid)
Pout_Proposed = np.array(Pout_Proposed)

# ===============================
# 4. 绘图
# ===============================
plt.figure(figsize=(8, 6))

plt.semilogy(CLWC, Pout_FSO, 'o-', lw=2, label='FSO–FSO Only')
plt.semilogy(CLWC, Pout_Hybrid, '>-', lw=2, label='Hybrid FSO/RF Only')
plt.semilogy(CLWC, Pout_Proposed, 's-', lw=2, label='Our Proposed System')

plt.xlabel('Cloud liquid water content (mg/m$^3$)')
plt.ylabel('Outage Probability')
plt.grid(True, which='both')
plt.legend()
plt.ylim([1e-10, 1])

plt.tight_layout()
plt.show()

# ===============================
# 5. 导出 Excel 表格
# ===============================
data = {
    'CLWC (mg/m^3)': CLWC,
    'Outage_FSO_FSO': Pout_FSO,
    'Outage_Hybrid_FSO_RF': Pout_Hybrid,
    'Outage_Proposed_System': Pout_Proposed
}

df = pd.DataFrame(data)
excel_filename = 'Outage_vs_CLWC.xlsx'
df.to_excel(excel_filename, index=False)

print(f'✅ Outage data saved to {excel_filename}')
