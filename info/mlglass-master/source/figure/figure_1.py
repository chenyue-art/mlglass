import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
df = pd.read_csv(r'F:\研一\论文\论文复现\论文4\info\mlglass-master\data\raw\TgTliqND300oxides.csv')

# 2. 去掉无用列
df = df.drop(columns=['Unnamed: 0'])

# 3. 计算每个样本的元素数量（非零元素）
element_cols = [col for col in df.columns if col not in ['Tg', 'ND300', 'Tliquidus']]
df['num_elements'] = (df[element_cols] > 0).sum(axis=1)

# 4. 熔化数据：变成三列：num_elements, property, value
df_melted = df.melt(
    id_vars=['num_elements'],
    value_vars=['Tg', 'ND300', 'Tliquidus'],
    var_name='property',
    value_name='value'
)

# 5. 设置对数变换（类似原图）
log_scale = {
    'Tg': False,
    'ND300': True,      # 假设 ND300 类似折射率
    'Tliquidus': False  # 你可以改为 True 如果需要
}

# 6. 绘图
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

properties = ['Tg', 'ND300', 'Tliquidus']

for ax, prop in zip(axes, properties):
    subset = df_melted[df_melted['property'] == prop].dropna()

    if log_scale[prop]:
        subset['value'] = np.log10(subset['value'])

    sns.histplot(
        data=subset,
        x='num_elements',
        weights='value',
        bins=range(3, 7),  # 假设最多6种元素
        ax=ax,
        kde=False,
        color='skyblue'
    )

    label = f"log₁₀({prop})" if log_scale[prop] else prop
    ax.set_xlabel("Number of chemical elements")
    ax.set_ylabel(f"Frequency ({label})")
    ax.set_title(f"{prop} Distribution")

plt.tight_layout()
plt.savefig("fig1_replica.png", dpi=300)
plt.show()