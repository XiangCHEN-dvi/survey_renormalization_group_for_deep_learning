# Survey: renormalization group for deep learning

本综述梳理**重整化群（Renormalization Group, RG）与深度学习、大语言模型（LLM）之间的理论与经验联系。RG 通过粗粒化（coarse-graining）与标度变换，保留“慢”自由度、积掉微观细节，从而解释临界现象与普适性；深度学习则通过层次结构在数据中逐层提取特征。二者在多尺度组织、幂律标度、相变与普适类**等概念上高度平行。

---

# Basics of Renormalization Group

## 核心思想

RG 源于统计物理与量子场论，用于研究系统在改变观测标度时的有效行为。典型流程包括：

1. **粗粒化**：将微观自由度合并为宏观变量（如 Kadanoff 块自旋）。
2. **标度/rescaling**：恢复原有“分辨率”，使理论形式不变。
3. **RG 流**：重复上述步骤，耦合常数在“理论空间”中形成流；**不动点**对应标度不变态。
4. **相关/无关算符**：在不动点附近，扰动按幂律放大或衰减；**临界指数**与**普适类**由不动点普适性质决定。

## 与机器学习的类比（为何相关）


| RG 概念 | 深度学习类比         |
| ----- | -------------- |
| 粗粒化   | 池化、降维、深层特征抽象   |
| 标度/深度 | 网络层数、感受野、上下文长度 |
| 不动点   | 训练收敛的表示、尺度不变特征 |
| 相变    | 性能突变、涌现能力、压缩临界 |
| 普适性   | 不同架构/数据上的共同标度律 |


## 经典 RG 方案（后文文献常引用）

- **实空间 RG（Kadanoff–Wilson）**：块自旋、积分掉短程涨落；Ising 模型是标准玩具系统。
- **Wilson 场论 RG**：动量空间积掉高能模式。
- **变分 RG**：在保留自由度的子空间上优化有效哈密顿量。
- **信息论 RG**：用互信息、相对熵等刻画哪些自由度应保留（见 [4] 及后续工作）。

---

# Renormalization Group for Deep Learning

本节按时间线综述：从“RG ↔ 深度学习”的概念类比，到严格映射、用神经网络**实现** RG，再到“深度学习是否等价于 RG 流”的实证检验。

## 概念桥梁：深度 = 标度，MERA ↔ 生成模型

### [1] Bény (2013) — *Deep learning and the renormalization group*

- **方法**：对比 RG 与深度学习中“深度/标度”的角色；将物理中的 **MERA（多尺度纠缠重整化拟设）** 改写为**层次生成式贝叶斯网络**，在局部关联假设下用显式概率（无需采样）学习分布。
- **观点**：RG 描述有效理论如何随观测标度变化；深度网络通过层次结构实现类似的多尺度组织。
- **结论**：MERA 可具体落地为一种深度学习算法；为后续“张量网络 / RG ↔ 深度生成模型”路线奠基（与 [2]、[6] 一脉相承）。

### [2] Mehta & Schwab (2014) — *An exact mapping between the variational renormalization group and deep learning*

- **方法**：建立 **Kadanoff 变分 RG** 与基于 **RBM 的深度网络** 之间的一一映射；在一维、二维最近邻 Ising 模型上数值验证。
- **观点**：DNN 可能在实现广义的、数据驱动的 RG 式粗粒化，以提取“相关特征”。
- **结论**：训练后的 RBM **自组织为类似块重整化的粗粒化**；这是 DL–RG 方向影响最大的严格结果之一。

### [3] Lin, Tegmark & Rolnick (2016) — *Why does deep and cheap learning work so well?*

- **方法**：用**信息论**形式化“廉价学习”：自然数据（尤其物理世界）往往具有**低内在维度、对称性、局域性、层次组合性**；深度网络用多项式/指数更少的参数逼近此类函数；证明若干 **no-flattening** 定理（浅层无法在不损失效率的情况下模拟深层）。
- **观点**：深度学习成功 partly 因为真实世界分布“像物理系统”一样可压缩；深度实现的是**组合性粗粒化**，与 RG 思想同构。
- **结论**：深度提供**标度上的计算效率**；与 RG 的联系在第三节（深度、重整化、复杂度）中系统讨论。

## 用机器学习做 RG：互信息、生成流、神经网络 RG

### [4] Koch-Janusz & Ringel (2017/2018, *Nature Physics*) — *Mutual Information, Neural Networks and the Renormalization Group*

- **方法**：定义**实空间互信息（RSMI）**；训练神经网络在**无先验物理知识**下识别空间区域间的相关信息，迭代执行 RG 粗粒化；应用于 1D/2D Ising 与二聚体模型。
- **观点**：RG 本质是信息保留问题——保留对宏观观测重要的自由度。
- **结论**：算法可复现 **RG 流**并提取 **Ising 临界指数**；证明 ML 可“学会”抽象物理概念，而不仅是拟合数据。

### [5] Iso, Shiba & Yokoo (2018, *Phys. Rev. E*) — *Scale-invariant Feature Extraction of Neural Network and Renormalization Group Flow*

- **方法**：用 **RBM** 学习多温度 Ising 自旋构型，构造参数（温度）随 RBM 迭代生成的**流**；辅以监督“温度计”网络估计有效温度。
- **观点**：DNN 特征提取可视为层次粗粒化，与 RG 类比。
- **结论**：无监督 RBM 流使有效温度**趋向临界值 T_c\approx 2.27**，方向与常规 RG（远离临界点）**相反**，体现**逆重整化/尺度不变特征提取**；固定点附近的构型具有近似尺度不变性。

### [6] Li & Wang (2018, *PRL*) — *Neural Network Renormalization Group*

- **方法**：基于 **normalizing flow** 的可逆生成模型，做**变分 RG**：物理空间 ↔ 互信息减小的潜空间；用**概率密度蒸馏**训练，损失给出自由能上界；在潜空间做 HMC 采样。
- **观点**：现代“信息保持 RG”可用深度生成模型实现；架构受 MERA 启发。
- **结论**：在 2D Ising 上识别出近似独立的集体变量，并加速采样；为“**NeuralRG**”——用 NN **正向实现** RG——的代表作。

## 深度学习是否“就是”RG 流？实证与严格对应

### [7] Koch et al. (2019) — *Is deep learning a renormalization group flow?*

- **方法**：在 Ising 数据上训练 **RBM**，将生成样本与 RG 处理后的构型对比；用**隐藏–可见神经元关联函数**诊断类 RG 粗粒化（主要比较**单层**与**单步 RG**）。
- **观点**：深度学习执行复杂的粗粒化，RG 可提供理论语言，但二者不必逐层等同。
- **结论**：关联函数中可见 **RG 样模式**，也存在**显著差异**；深度学习 **≠** 标准 RG 流的简单复制。

### [8] Haggi Mani (2022, 硕士论文) — *Renormalization group theory, scaling laws and deep learning*

- **方法**：综述性工作：格点 RG、Ising/Hopfield/RBM、统计场论与量子场论工具；联系神经网络的**幂律标度、相变、算法普适性**。
- **观点**：RG 可能是解释深度学习经验规律（标度律、相变式行为）的候选统一框架。
- **结论**：系统梳理物理与 ML 的对应，强调**标度律起源**与**算法普适性**；偏理论综合，非单一新算法。

### [9] Gong & Xia (2022) — *Interpreting deep learning by establishing a rigorous corresponding relationship with renormalization group*

- **方法**：对**全连接网络**与**一维 Ising 实空间 RG** 建立严格对应；证明当网络参数达到特定条件时，输出耦合常数的极限等于 RG **不动点**处的耦合常数。
- **观点**：在受控设定下，**训练过程在数学上等价于 RG**。
- **结论**：网络从微观自旋构型中提取**宏观有效理论**；为 DNN 可解释性提供物理严格框架（限于 1D Ising + 特定架构）。

### [10] Taylor (2023) — *A Deep Dive into the Connections Between the Renormalization Group and Deep Learning in the Ising Model*

- **方法**：系统实现 1D/2D Ising 的 **Kadanoff 块 RG**（含 Wolff 采样、临界指数 \nu 校验）；逐层分析 **RBM** 学习，并与自旋重整化权重直接对比。
- **观点**：无监督深度学习可能是**逐层粗粒化的 RG 流**（延续 [2][5] 等）。
- **结论**：RBM 学习呈现与 RG **定性相似的 blocking 结构**，但与最近邻 Ising 自旋重整化的**定量不一致**；类比成立但需审慎推广。

## 现代理论：标度律、普适性与学习曲线的 RG

### [15] Coppola, Helias & Ringel (2025) — *Renormalization group for deep neural networks: Universality of learning and scaling laws*

- **方法**：对**弱非线性（非 lazy）网络在幂律分布数据**上的学习曲线建立 RG 框架；考虑核谱离散、无平移不变性等 NN 场论特点。
- **观点**：数据与模型中的自相似性可用 RG 分析；传统**标度维**需推广为**标度区间（scaling intervals）**。
- **结论**：扰动仍可分为 relevant/irrelevant；大数据极限下存在 **GP 型 UV 不动点**及普适性（类渐近自由）；为神经缩放律提供场论式基础。

---

# Renormalization Group for Large Language Models

LLM 文献较少直接实现 Wilson RG，更多借用**相变、临界性、粗粒化、多尺度记忆**等 RG 语言描述**输出分布、推理、压缩与涌现**。

## 输出分布与采样温度：自动探测相图

### [11] Arnold et al. (2024) — *Phase transitions in the output distribution of large language models*

- **方法**：将物理中**自动相变检测**（基于数据的统计距离）适配到 LLM；用 **f-散度**度量**下一 token 分布**随控制参数（温度、训练步数、提示等）的变化；在 Pythia、Mistral、Llama 等上实验。
- **观点**：LLM 在参数变化时可出现类似物相的**分布突变**。
- **结论**：无需人工选定序参量即可绘制“相图”；可发现新行为相与未预期的转变，适用于快速迭代的生成模型分析。

### [12] Nakaishi, Nishikawa & Hukushima (2024) — *Critical phase transition in large language models*

- **方法**：对 **GPT-2** 生成文本做统计物理分析：关联长度、磁化类序参量、谱性质等随**采样温度 T** 变化。
- **观点**：低温（有序、重复）与高温（无序、不可读）之间的质变是否为**真相变**。
- **结论**：在 **T_c\approx 1** 附近出现**发散/临界行为**（幂律关联衰减、慢弛豫等），与自然语言语料相似；可理解性可能与**临界区瞬态**有关；为“语言 ≈ 临界系统”提供数值证据。

### [13] Sun & Haghighat (2025) — *Phase transitions in large language models and the O(N) model*

- **方法**：将 **Transformer** 重写为 **O(N) 场论模型**；分析生成温度与参数量 P 两个控制量。
- **观点**：LLM 丰富标度行为可用场论相变语言理解。
- **结论**：(1) **温度相变** → 可估计模型**内禀维度**；(2) **参数量相变**（“相变的相变”）在 **P_c\approx 7B** 附近，大模型与小模型行为质异（涌现）；提出用 **O(N) 能量** 评估训练/参数是否“够大”。

## RG 启发的系统架构：多尺度记忆

### [14] Tian et al. (2025) — *RGMem: Renormalization group-based memory evolution for language agent user profile*

- **方法**：**RGMem**：将对话记忆视为多尺度过程——情节 → 语义事实 → 用户洞察，经**层次粗粒化、阈值更新、rescaling** 汇入动态用户画像；区分快变证据与慢变特质。
- **观点**：长期个性化需要 RG 式的**信息压缩与涌现**，而非扁平 RAG。
- **结论**：在 LOCOMO、PersonaMem 上优于 SOTA 记忆系统（约 +7~9 分）；证明 RG 隐喻可工程化为**语言智能体记忆演化**。

## 压缩、推理与涌现：临界性与“相变”隐喻

### [16] Ma et al. (2026, *npj AI*) — *Phase transitions in large language model compression*

- **方法**：提出 **Model Phase Transition (MPT)**：系统评测 30+ 剪枝/量化/低秩方法；用**分段幂律–指数**拟合性能–压缩曲线，标定**相变点（PTP）**。
- **观点**：冗余分**结构、数值、代数**三类，且**正交**；联合压缩应在各 PTP 围成的“安全区”内规划轨迹。
- **结论**：单方法 PTP 例：非结构化剪枝 ~65% 稀疏、结构化 ~45%、量化 ~3-bit、低秩 ~30% 秩保留；**联合策略下可近无损压缩至约 10% 原始规模**；“压缩巨人优于训练矮人”。

### [17] Almaghrabi (2026) — *Phase transitions in large language model reasoning: A stochastic framework for critical compute thresholds in test-time scaling*

- **方法**：将推理建模为**随机分支搜索**（每步正确率 p、分支因子 b、深度 d、算力预算 C）；大规模 Monte Carlo；定义临界算力、转变宽度、susceptibility 等（与作者 viXra:2603.0024 等预印本一脉相承）。
- **观点**：测试时算力（CoT、自一致性、树搜索）下，性能可能呈**相变式阈值**而非平滑增长；bp 为控制正确路径能否增殖的关键参量。
- **结论**：算力略过临界区则成功率陡升；改进推理需同时提高**单步可靠性**或**有效分支**；为 test-time scaling 提供统计物理式概念框架（简化模型，非 LLM 内部机制证明）。

### [18] Krakauer, Mitchell & Krakauer (2026, *Phil. Trans. R. Soc. A*) — *Large language models and emergence: A complex systems perspective*

- **方法**：从复杂性科学梳理**涌现**的严格条件：**标度、临界性、压缩、新基、泛化**；区分 **knowledge-out**（简单组分）与 **knowledge-in**（复杂环境/数据塑造，LLM 属此类）；区分**涌现能力** vs **涌现智能**（“more is different” vs “less is more”）。
- **观点**：LLM 文献常把基准上的**突变或意外能力**称为涌现，但这不足以构成科学意义上的涌现；需存在**因果充分的粗粒化描述**（有效理论）屏蔽微观细节。
- **结论**：LLM 最多展示**涌现能力**（规模带来的新任务表现），尚难论证**涌现智能**（高效类比与“以少驭多”）；与 RG/相变叙事对接时，应区分**现象学 S 形曲线**与**具有粗粒化机制的真涌现**。

## 教材路线：统计物理 → RG → 神经网络 → LLM

### [19] Hohm (2026) — *Lecture Notes on Statistical Physics and Neural Networks*

- **方法**：洪堡大学硕士统计物理课讲义（约课程 30–40%）；自洽引入 Boltzmann–Gibbs、Ising/自旋玻璃与**热力学极限相变**；**RG** 表述为对自由度的 **integrating out / 边缘化** 及耦合常数流；Hopfield 网络与 BM/RBM 学习（隐藏神经元积掉与 RG 对照，见 [2][6][7]）；前馈网络、**反向传播**、**Transformer/LLM** 入门；展望 **double descent** 与 **Kaplan 缩放律**。
- **观点**：统计物理是理解 NN/DL 的**低门槛入口**（相较 QFT）；RBM 训练中“积掉隐藏层”与 Ising RG **同语言、未必同格点标度**；LLM 自回归中 prompt token 像**变耦合常数**，逐步预测近似 **RG 流** 的隐喻；LLM 性能幂律与临界**标度律/普适性**类比，但数值普适性弱于物理临界现象。
- **结论**：**非新研究**，而是把本综述多条脉络串成**可读教材**；强调需区分定量经验规律（泛化曲线、\(L(N,D,C)\)）与尚待建立的 DL“热力学”； cites 本表多篇文献（含 [2][4][6][7][13][15] 等）。

---

## 脉络小结与文献分类

下表合并脉络归类、网络结构、RG–DL 关系类型与要点说明。关系类型列直接写出归类内容；**加粗**表示该文核心贡献方向。

| 编号 | 神经网络 / 模型结构 | 关系类型 | 脉络方向 | 关系强度 | 说明 |
|------|-------------------|---------|---------|---------|------|
| [1] | MERA 启发的层次生成式贝叶斯网络 | **用 RG 训练/学习** + **RG 启发的网络/系统结构** + 理论/概念类比 | 类比/实证检验 | 中（部分一致） | MERA 改写为可训练生成模型；深度层次对应多尺度 RG |
| [2] | RBM 堆叠深度网络 | **用 RG 分析神经网络** + 理论/概念类比 | 严格等价/映射 | 强（特定模型） | 变分 RG 与 RBM-DNN 严格映射；训练自组织为块粗粒化 |
| [3] | 一般浅层/深层网络（理论） | 理论/概念类比 | 学习理论/标度律 | 中–强（理论） | 廉价学习、no-flattening；深度效率与 RG/组合性类比 |
| [4] | RSMI 优化用 ANN | **深度学习用于 RG/统计物理** | NN 实现 RG | 强（物理系统） | 网络迭代执行 RG 粗粒化；Ising/二聚体，提取临界指数 |
| [5] | RBM + 监督温度计小网络 | **用 RG 分析神经网络** | 类比/实证检验 | 中（部分一致） | RBM 流 vs Ising RG；逆流向临界点 \(T_c\approx 2.27\) |
| [6] | Normalizing flow（realNVP，MERA 式层次） | **深度学习用于 RG/统计物理** + **RG 启发的网络/系统结构** | NN 实现 RG | 强（物理系统） | NeuralRG 变分粗粒化 Ising；可逆流 + 概率密度蒸馏 |
| [7] | RBM（无监督，Ising） | **用 RG 分析神经网络** | 类比/实证检验 | 中（部分一致） | 单层 RBM vs 单步 RG；关联函数见 RG 样模式亦有差异 |
| [8] | Hopfield、RBM 等（综述） | 理论/概念类比 | 学习理论/标度律 | 中–强（理论） | 综述：标度律、相变与 DL 的 RG 理论联系 |
| [9] | 全连接 DNN（1D Ising） | **用 RG 分析神经网络** | 严格等价/映射 | 强（特定模型） | 特定条件下训练严格等价于 1D 实空间 RG |
| [10] | RBM（与 Ising RG 对照） | **用 RG 分析神经网络** | 类比/实证检验 | 中（部分一致） | Kadanoff RG 基线；与块 RG 定性相似、定量不一致 |
| [11] | Transformer LLM（Pythia/Mistral/Llama） | **统计物理/相变分析（非显式 RG 流）** | LLM 相变现象 | 隐喻+统计工具 | f-散度自动检测输出分布相变 |
| [12] | GPT-2 | **统计物理/相变分析（非显式 RG 流）** | LLM 相变现象 | 隐喻+统计工具 | 采样温度 \(T_c\approx 1\) 附近文本临界相变 |
| [13] | Transformer（\(O(N)\) 场论重写） | **统计物理/相变分析（非显式 RG 流）** | LLM 相变现象 | 隐喻+统计工具 | 温度与参数量相变；\(P_c\approx 7\)B「相变的相变」 |
| [14] | LLM 智能体 + RGMem 记忆工作流 | **RG 启发的网络/系统结构** | RG 启发工程 | 应用架构 | 多尺度记忆粗粒化；不改 Transformer 层 |
| [15] | 弱非线性非 lazy 特征学习网络（理论） | **用 RG 分析神经网络** | 学习理论/标度律 | 中–强（理论） | RG 分析学习曲线；标度区间与 GP UV 不动点 |
| [16] | 各类 LLM（压缩评测） | **统计物理/相变分析（非显式 RG 流）** | LLM 相变现象 | 隐喻+统计工具 | Model Phase Transition；压缩–性能相变点 |
| [17] | 无（随机分支搜索模型） | **抽象框架（无/弱绑定具体 NN）** | LLM 相变现象 | 隐喻+统计工具 | test-time 算力临界阈值；Monte Carlo 推理树 |
| [18] | LLM（概念层面） | **理论/概念类比** + **抽象框架（无/弱绑定具体 NN）** | 学习理论/标度律 | 中–强（理论） | 涌现的粗粒化条件；涌现能力 vs 涌现智能 |
| [19] | Hopfield/BM/RBM、前馈网络、Transformer/LLM（教材） | **理论/概念类比** + 教材/综述 | 类比/教材串联 | 中（入门） | 统计物理→RG→NN→LLM 一条龙；RBM 积掉隐藏层≈RG；缩放律 outlook |

---

## References

[1] Bény, Cédric. "Deep learning and the renormalization group." arXiv preprint arXiv:1301.3124 (2013).

[2] Mehta, Pankaj, and David J. Schwab. "An exact mapping between the variational renormalization group and deep learning." arXiv preprint arXiv:1410.3831 (2014).

[3] Lin, Henry W., Max Tegmark, and David Rolnick. "Why does deep and cheap learning work so well?." arXiv preprint arXiv:1608.08225 (2016).

[4] Koch-Janusz, Maciej, and Zohar Ringel. "Mutual Information, Neural Networks and the Renormalization Group." arXiv preprint arXiv:1704.06279 (2017).

[5] Isoa, Satoshi, Shotaro Shibaa, and Sumito Yokooa. "Scale-invariant Feature Extraction of Neural Network and Renormalization Group Flow." arXiv preprint arXiv:1801.07172 (2018).

[6] Li, Shuo-Hui, and Lei Wang. "Neural Network Renormalization Group." arXiv preprint arXiv:1802.02840 (2018).

[7] Koch, Ellen De Mello, Robert De Mello Koch, and Ling Cheng. "Is deep learning a renormalization group flow?." arXiv preprint arXiv:1906.05212 (2019).

[8] Haggi Mani, Parviz. "Renormalization group theory, scaling laws and deep learning." (2022).

[9] Gong, Fuzhou, and Zigeng Xia. "Interpreting deep learning by establishing a rigorous corresponding relationship with renormalization group." arXiv preprint arXiv:2212.00005 (2022).

[10] Taylor, Kelsie. "A Deep Dive into the Connections Between the Renormalization Group and Deep Learning in the Ising Model." arXiv preprint arXiv:2308.11075 (2023).

[11] Arnold, Julian, et al. "Phase transitions in the output distribution of large language models." arXiv preprint arXiv:2405.17088 (2024).

[12] Nakaishi, Kai, Yoshihiko Nishikawa, and Koji Hukushima. "Critical phase transition in large language models." arXiv preprint arXiv:2406.05335 (2024).

[13] Sun, Youran, and Babak Haghighat. "Phase Transitions in Large Language Models and the O(N) Model." arXiv preprint arXiv:2501.16241 (2025).

[14] Tian, Ao, et al. "Rgmem: Renormalization group-based memory evolution for language agent user profile." arXiv preprint arXiv:2510.16392 (2025).

[15] Coppola, Gorka Peraza, Moritz Helias, and Zohar Ringel. "Renormalization group for deep neural networks: Universality of learning and scaling laws." arXiv preprint arXiv:2510.25553 (2025).

[16] Ma, Ziyang, et al. "Phase transitions in large language model compression." npj Artificial Intelligence 2.1 (2026): 21.

[17] Almaghrabi, Sif. "Phase Transitions in Large Language Model Reasoning: A Stochastic Framework for Critical Compute Thresholds in Test-Time Scaling." (2026).

[18] Krakauer, David C., Melanie Mitchell, and John W. Krakauer. "Large language models and emergence: A complex systems perspective." Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 384.2320 (2026): 20250014.

[19] Hohm, Olaf. "Lecture Notes on Statistical Physics and Neural Networks." arXiv preprint arXiv:2605.06394 (2026).