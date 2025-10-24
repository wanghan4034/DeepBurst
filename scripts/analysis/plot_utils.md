Thank you for providing the additional guidance! Based on your input, here’s a refined Python Matplotlib configuration and plotting example that adheres to **Nature**'s figure preparation guidelines, including the specific points you mentioned.

---
要符合《Science》期刊的图表要求，需要特别注意图表的清晰度、字体、线宽、颜色、分辨率等细节。《Science》期刊对图表的格式有明确的要求，以下是基于《Science》官方指南的Python Matplotlib配置示例，帮助你生成符合要求的图表。

---

### 《Science》期刊图表要求的关键点：
1. **字体**：
   - 使用 **Sans-serif** 字体（如 **Arial** 或 **Helvetica**）。
   - 字体大小：坐标轴标签 **8-10 pt**，图例和标题 **10-12 pt**。

2. **线宽**：
   - 线条宽度：**1-1.5 pt**。
   - 坐标轴线宽：**1 pt**。

3. **颜色**：
   - 避免使用纯红色和纯绿色（色盲友好）。
   - 使用高对比度的颜色。

4. **分辨率**：
   - 分辨率至少为 **300 DPI**。
   - 推荐保存为 **TIFF** 或 **PDF** 格式。

5. **尺寸**：
   - 单栏图宽度：**8.7 cm**。
   - 双栏图宽度：**18 cm**。
   - 高度通常不超过宽度。

6. **图例**：
   - 图例应清晰，避免遮挡数据。
   - 图例字体大小与坐标轴标签一致。

---

### Python Matplotlib 配置示例

以下是符合《Science》期刊要求的 Matplotlib 配置和绘图代码：

```python
import matplotlib.pyplot as plt
import numpy as np

# 设置全局样式以符合《Science》期刊要求
plt.rcParams.update({
    'font.family': 'sans-serif',  # 使用Sans-serif字体
    'font.sans-serif': ['Arial', 'Helvetica'],  # 指定Arial或Helvetica字体
    'font.size': 10,  # 全局字体大小
    'axes.labelsize': 10,  # 坐标轴标签字体大小
    'axes.titlesize': 12,  # 标题字体大小
    'xtick.labelsize': 8,  # x轴刻度标签字体大小
    'ytick.labelsize': 8,  # y轴刻度标签字体大小
    'legend.fontsize': 8,  # 图例字体大小
    'figure.titlesize': 12,  # 图形标题字体大小
    'lines.linewidth': 1.5,  # 线宽
    'axes.linewidth': 1.0,  # 坐标轴线宽
    'xtick.major.width': 1.0,  # x轴主刻度线宽
    'ytick.major.width': 1.0,  # y轴主刻度线宽
    'xtick.minor.width': 0.5,  # x轴次刻度线宽
    'ytick.minor.width': 0.5,  # y轴次刻度线宽
    'xtick.major.size': 5,  # x轴主刻度长度
    'ytick.major.size': 5,  # y轴主刻度长度
    'xtick.minor.size': 3,  # x轴次刻度长度
    'ytick.minor.size': 3,  # y轴次刻度长度
    'figure.dpi': 300,  # 图形分辨率
    'savefig.dpi': 300,  # 保存图形时的分辨率
    'savefig.format': 'tiff',  # 保存为TIFF格式
    'savefig.bbox': 'tight',  # 保存图形时裁剪空白区域
    'figure.autolayout': True,  # 自动调整图形布局
})

# 创建示例数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(8.7/2.54, 6/2.54))  # 设置图形大小（单位：英寸，8.7 cm为单栏宽度）

# 绘图
ax.plot(x, y1, label='sin(x)', color='blue', linewidth=1.5)  # 使用蓝色
ax.plot(x, y2, label='cos(x)', color='orange', linewidth=1.5)  # 使用橙色

# 设置坐标轴标签
ax.set_xlabel('x-axis', fontsize=10)
ax.set_ylabel('y-axis', fontsize=10)

# 设置标题
ax.set_title('Example Plot for Science Journal', fontsize=12)

# 添加图例
ax.legend(frameon=False)  # 去掉图例边框

# 设置刻度
ax.tick_params(axis='both', which='major', width=1.0, size=5)
ax.tick_params(axis='both', which='minor', width=0.5, size=3)

# 显示图形
plt.show()

# 保存图形
fig.savefig('science_figure.tiff', dpi=300)
```

---

### 关键配置说明：
1. **字体**：
   - 使用 `Arial` 或 `Helvetica`，这是《Science》推荐的字体。
   - 字体大小设置为 8-10 pt，标题可以稍大（12 pt）。

2. **图形尺寸**：
   - 单栏图宽度为 8.7 cm（转换为英寸：8.7 / 2.54 ≈ 3.43 英寸）。
   - 高度通常为宽度的一半到三分之二。

3. **线宽和刻度**：
   - 线条宽度设置为 1.5 pt，坐标轴线宽为 1 pt。
   - 刻度线宽和长度也符合《Science》的要求。

4. **颜色**：
   - 使用高对比度的颜色（如蓝色、橙色），避免红色和绿色。

5. **保存格式**：
   - 保存为 TIFF 格式，分辨率为 300 DPI。

---

### 输出结果：
运行上述代码后，生成的图表将符合《Science》期刊的要求，可以直接用于投稿。如果需要双栏图，只需调整 `figsize` 的宽度为 18 cm（转换为英寸：18 / 2.54 ≈ 7.09 英寸）。

如果有其他具体要求（如颜色映射、误差条等），可以进一步调整代码。

---
### Key Adjustments for Nature's Guidelines:
1. **Lettering in Figures**:
   - Use **lower-case type** for axis labels, with the **first letter capitalized**.
   - Avoid full stops in labels.

2. **Units**:
   - Add a **single space** between numbers and units (e.g., `1.5 m`).
   - Use **SI units** or field-specific nomenclature.
   - Separate thousands with commas (e.g., `1,000`).

3. **Scale Bars**:
   - Use **scale bars** instead of magnification factors (for images or plots with physical dimensions).

4. **Text Placement**:
   - Avoid placing text directly over shaded or textured areas.
   - Avoid reversed type (white text on colored background) unless necessary.
   - Place text, including keys to symbols, in the **legend** rather than on the figure itself.

5. **General Styling**:
   - Use **sans-serif fonts** (e.g., Arial or Helvetica).
   - Keep lines thin and precise (e.g., 0.5–0.75 pt for data lines).

---

### Updated Python Matplotlib Code

Here’s an example that incorporates all the above guidelines:

```python
import matplotlib.pyplot as plt
import numpy as np

# Set global style to match Nature's guidelines
plt.rcParams.update({
    'font.family': 'sans-serif',  # Use sans-serif font
    'font.sans-serif': ['Arial', 'Helvetica'],  # Specify Arial or Helvetica
    'font.size': 8,  # Base font size
    'axes.labelsize': 8,  # Axis label font size
    'axes.titlesize': 10,  # Title font size
    'xtick.labelsize': 7,  # X-axis tick label size
    'ytick.labelsize': 7,  # Y-axis tick label size
    'legend.fontsize': 8,  # Legend font size
    'figure.titlesize': 10,  # Figure title size
    'lines.linewidth': 0.75,  # Line width for data
    'axes.linewidth': 0.5,  # Axis line width
    'xtick.major.width': 0.5,  # X-axis major tick width
    'ytick.major.width': 0.5,  # Y-axis major tick width
    'xtick.minor.width': 0.25,  # X-axis minor tick width
    'ytick.minor.width': 0.25,  # Y-axis minor tick width
    'xtick.major.size': 4,  # X-axis major tick length
    'ytick.major.size': 4,  # Y-axis major tick length
    'xtick.minor.size': 2,  # X-axis minor tick length
    'ytick.minor.size': 2,  # Y-axis minor tick length
    'figure.dpi': 300,  # Figure resolution
    'savefig.dpi': 300,  # Save resolution
    'savefig.format': 'tiff',  # Save format (TIFF recommended)
    'savefig.bbox': 'tight',  # Tight bounding box
    'figure.autolayout': True,  # Automatically adjust layout
})

# Create example data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y1_err = 0.1 * np.random.rand(100)  # Error data

# Create figure and axis
fig, ax = plt.subplots(figsize=(8.7/2.54, 6/2.54))  # Single-column width (8.7 cm)

# Plot data
ax.plot(x, y1, label='Sin(x)', color='blue', linewidth=0.75)  # Thin lines
ax.plot(x, y2, label='Cos(x)', color='orange', linewidth=0.75)

# Add error bars
ax.errorbar(x[::10], y1[::10], yerr=y1_err[::10], fmt='o', color='blue', capsize=3, capthick=0.5, elinewidth=0.5)

# Set axis labels (lower-case, first letter capitalized)
ax.set_xlabel('Time (s)', fontsize=8)  # Units with space
ax.set_ylabel('Amplitude (m)', fontsize=8)

# Set title
ax.set_title('Example Plot for Nature Journal', fontsize=10)

# Add legend (place keys here instead of on the figure)
ax.legend(frameon=False, loc='upper right')

# Set ticks and grid
ax.tick_params(axis='both', which='major', width=0.5, size=4)
ax.tick_params(axis='both', which='minor', width=0.25, size=2)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

# Add scale bar (example for images)
# For plots, scale bars are less common, but you can use annotations if needed
ax.annotate('1 s', xy=(1, -1.2), xycoords='data', fontsize=7, ha='center', va='center')

# Save and show
plt.show()
fig.savefig('nature_figure.tiff', dpi=300)
```

---

### Key Features in the Code:
1. **Lettering**:
   - Axis labels are in lower-case with the first letter capitalized (e.g., `Time (s)`).
   - No full stops are used in labels.

2. **Units**:
   - Units are separated from numbers by a single space (e.g., `Amplitude (m)`).

3. **Scale Bar**:
   - A scale bar is added as an annotation (for demonstration purposes; adjust as needed).

4. **Text Placement**:
   - All text is placed in the legend or axis labels, avoiding direct placement over shaded areas.
   - No reversed type is used.

5. **Styling**:
   - Thin lines (0.75 pt) and precise ticks (0.5 pt) are used.
   - A grid is added for clarity, with a subtle dashed style.

---

### Output:
The resulting figure will be:
- Saved as a **300 DPI TIFF** file.
- Formatted for **single-column width** (8.7 cm).
- Compliant with **Nature**'s guidelines for lettering, units, and styling.

Let me know if you need further adjustments!