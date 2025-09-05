# SEKI：**Symmetry-Equivariant Karcher Interpolation** — 在資訊幾何上的流形鄰域資料增強

一句話摘要：把資料點放回它們真正「活著」的流形上，沿著**測地線**做等變插值，形成更貼近真實資料流形的鄰域分佈（VRM），以降低幾何失真與標註混亂，提升泛化與魯棒性。SEKI 同時結合**群對稱等變性**與**Karcher/Fréchet 幾何平均**來保形增強，理論上立基於資訊幾何（Fisher–Rao）與黎曼幾何工具。([papers.neurips.cc][1], [scholarpedia.org][2], [content.e-bookshelf.de][3])

---

## 研究核心問題

* 如何在**非歐幾里得**資料空間（如機率分佈流形、SPD 矩陣流形、球面/超曲率流形）中，構造**幾何一致**且**對稱等變**的資料增強？
* 是否能把 **VRM（Vicinal Risk Minimization）** 的想法從歐式線性插值推廣到**測地線插值**，藉此更貼近真實資料流形，並在理論上獲得更好的近似與穩定性？([papers.neurips.cc][1])

## 研究目標

1. 提出 SEKI：於流形上使用 **Karcher/Fréchet 插值**（經 exponential/log 對映）生成 vicinal 樣本，並**強化對稱等變性**（旋轉/反射等群作用）。
2. 在**資訊幾何**（Fisher–Rao 度量）與**SPD 流形**（Log-Euclidean）兩大場景給出**統一的算法與理論分析**。([scholarpedia.org][2], [PubMed][4])
3. 建立 **VRM 在流形設定**下的**風險分解/界限**與**穩定性**結果。([papers.neurips.cc][1])
4. 在標準/合成資料與 sklearn toy datasets 驗證效益，同時與 Mixup/Manifold Mixup/R-Mixup、G-CNN 等做系統比較。([arXiv][5], [Proceedings of Machine Learning Research][6])

## 方法總覽（SEKI Pipeline）

* **幾何選擇**：

  * **統計流形**：以 Fisher–Rao 度量表述機率分佈的內在幾何（例如類別條件分佈或輸出分佈），用自然梯度觀點輔助優化與穩健性分析。([scholarpedia.org][2], [dl.acm.org][7], [jmlr.org][8])
  * **SPD 流形**：對協方差/圖結構資料用 **Log-Euclidean** 幾何，計算快且避免「swelling effect」。([PubMed][4])
* **等變性處理**：選定群 $G$（如旋轉、反射），在插值前先將資料對齊至同一群軌道代表或在群作用後做**軌道平均**，確保生成樣本對模型/資料保持**等變性**。([Proceedings of Machine Learning Research][9])
* **Karcher/Fréchet 測地線插值**：給定 $x_i, x_j$，以 $\tilde x=\exp_{x_i}\!\big(\lambda\,\log_{x_i}(g\!\cdot\!x_j)\big)$ 生成流形上的「線性」樣本；標籤以 geodesic 長度配權（或在機率流形上用 Fisher-Rao geodesic 插值）。([vision.jhu.edu][10], [jstor.org][11], [arXiv][12])
* **Vicinal 風險**：以流形 vicinal 分佈最小化風險，對照歐式 Mixup 的線性凸組合。([papers.neurips.cc][1], [arXiv][5])

## 預期貢獻

1. **方法**：首個同時滿足**流形幾何一致**與**群等變**的測地線資料增強框架。
2. **理論**：給出在**有界曲率/注入半徑**條件下，SEKI 相對於歐式插值在 vicinal 風險近似誤差與穩定性的**上界**；建立 Fisher–Rao 幾何下的**自然梯度-一致性**連結。([dl.acm.org][7])
3. **實務**：於 SPD 與 toy 流形資料上展現比 Mixup/Manifold Mixup 更穩定的**校準度（ECE）**與**對稱變換魯棒性**；在圖/生醫 SPD 場景中超越 **R-Mixup**。([arXiv][13])
4. **開源**：提供 Pytorch/NumPy 參考實作與幾何介面（Exp/Log、Karcher 平均、群作用）。

## 創新

* 將 **VRM** 從歐式線段推進到**流形測地線**，避免歐式插值對曲率的失真。([papers.neurips.cc][1])
* 在增強層面引入**等變性約束**（與 G-CNN 層級等變形成互補），用**群軌道/商空間**思想降低不必要的資料多樣性。([Proceedings of Machine Learning Research][9])
* 以 **Karcher/Fréchet 幾何平均**與 **Log-Euclidean** 快速計算結合，兼顧理論正當性與效率。([vision.jhu.edu][10], [PubMed][4])
* 與 **資訊幾何/自然梯度**接軌，將優化與資料增強的幾何基礎統一化。([dl.acm.org][7])

## 理論洞見（將證明/推導）

1. **流形-VRM 風險界**：在截斷測地凸域、曲率界 $|K|\le \kappa$ 下，SEKI 所誘發的 vicinal 風險與真實風險的差距較歐式 mixup 更小（差距中含曲率與測地偏差項）。思路承接 VRM 的泛化分析並加上比較幾何工具。([papers.neurips.cc][1])
2. **等變穩定性**：若資料與標籤規則對 $G$ 等變，則 SEKI 生成分佈對 $G$ 不變，從而降低由無關變換造成的方差；此與 G-CNN 的結構性等變互補。([Proceedings of Machine Learning Research][9])
3. **自然梯度對齊**：在 Fisher–Rao 幾何下，SEKI 的測地線插值與自然梯度的**最速下降方向**一致性提升模型訓練的幾何相容性。([dl.acm.org][7])
4. **Karcher 均值收斂/唯一性條件**：使用 Afsari 與後續工作對 Riemannian 中心/均值的存在唯一與收斂性結果作為理論基石。([vision.jhu.edu][10], [jstor.org][11])

## 數學理論推演與證明路線（提要）

* **定義**：在流形 $(\mathcal M, g)$ 與群作用 $G$ 下，定義 SEKI vicinal 分佈 $\nu_{\text{SEKI}}$ 與風險 $R_\nu$。
* **幾何工具**：用 **Exp/Log** 對映定義測地線插值；引入 **Karcher 功能** 的強凸性條件；以 **Jacobi 場** 或比較定理界定測地偏差。([vision.jhu.edu][10])
* **VRM 一般化**：改寫 Chapelle 等的 VRM 分解，將「鄰域」由球/凸組合換成測地球/測地線段。([papers.neurips.cc][1])
* **等變性命題**：證明在群作用等變與測地相容條件下，$\nu_{\text{SEKI}}$ 對 $G$ 不變。([Proceedings of Machine Learning Research][9])
* **Fisher–Rao 場景**：在部分分佈族（如高斯/Dirichlet）使用已知 **Fisher–Rao 距離或閉式表達**簡化證明與估計。([PMC][14], [arXiv][12])
* **SPD 場景**：借助 **Log-Euclidean** 幾何把插值化為矩陣對數域的歐式計算，並連結 R-Mixup 的特殊情形。([PubMed][4], [arXiv][13])

## 實驗與資料集

* **合成/Sklearn toy**

  * `make_moons`, `make_circles`, `make_swiss_roll`（含旋轉/反射群作用）作為低維流形/曲率測試牆。([scikit-learn.org][15])
  * `make_spd_matrix` 生成 SPD 样本（可再施加群作用如同時相似變換）。([scikit-learn.org][16])
* **基線**：ERM、Mixup、Manifold Mixup、R-Mixup（SPD 場景）、資料層級與架構層級的等變方法（G-CNN）。([arXiv][5], [Proceedings of Machine Learning Research][6])
* **評估**：準確率、ECE 校準度、對群變換魯棒性（test-time augmentation）、小樣本/標注噪聲、輕量 OOD 轉移。
* **消融**：曲率（資料幾何）vs. SEKI 插值深度、等變處理（前/後對齊）、Fisher–Rao vs. Log-Euclidean。

## 成功投稿計畫

* **第一波**：完整理論 + toy/中小型資料實驗 → 幾何/統計導向會議（AISTATS / ALT）或 **ICLR/NeurIPS 幾何工作坊**。
* **擴充版**：加上影像（旋轉 MNIST、簡化 CIFAR 的群等變測試）與更強理論 → **ICLR/NeurIPS 主會**。
* **材料**：開源程式碼、幾何可視化、可復現腳本與數學附錄（含 Exp/Log、群作用、Karcher 收斂假設）。

## 失敗投稿備案（降階路線）

* 若主結果有限或只在 SPD 場景顯著：

  * 針對 SPD/醫療影像/圖網路的專領域研討會或期刊專欄；
  * 以「**R-Mixup 的一致化與等變化**」為短文，聚焦效率與穩定性對比。([arXiv][13])
* 先行上傳 arXiv，收集回饋後加強證明與消融，再投工作坊或次年主會。

## 與現有研究之區別

* **相較 Mixup / Manifold Mixup**：它們在**歐式空間**線性插值（輸入或隱表示），SEKI 則在**黎曼流形**上沿**測地線**插值，並**顯式納入群等變性**；理論上處理曲率項並以 VRM 重新推導。([arXiv][5], [Proceedings of Machine Learning Research][6])
* **相較 R-Mixup（SPD）**：R-Mixup 採 **Log-Euclidean** 對 SPD 做插值；SEKI 把此視為**特例**，並將框架推廣至任意（含 Fisher–Rao）流形，同時加入**群等變**與**統一理論**。([arXiv][13])
* **相較 G-CNN**：G-CNN 在**模型結構**上等變；SEKI 在**資料分佈**上等變，兩者可疊加，從資料與網路兩邊共同減少樣本複雜度。([Proceedings of Machine Learning Research][9])
* **幾何平均/中心**：SEKI 依賴 Karcher/Fréchet 幾何平均與其收斂/唯一性條件，這在既有增強法中少見。([vision.jhu.edu][10], [jstor.org][11])
* **資訊幾何連結**：把 Fisher–Rao 與自然梯度的理論優勢引入 data augmentation 的設計與分析中。([dl.acm.org][7])

---

## 可能的主要定理與證明草案（可寫入論文）

* **定理 A（流形-VRM 逼近）**：在測地凸域內、損失 $L$ 為 $\beta$-Lipschitz，曲率界 $|K|\le\kappa$ 時，SEKI-VRM 的結構性偏差項包含 $\mathcal O(\kappa \cdot \text{(弦長)}^3)$ 的幾何誤差，而歐式 Mixup 的誤差上界額外含曲率-不匹配項。
* **定理 B（等變性不變風險）**：若資料與標籤規則對 $G$ 等變，則 SEKI 誘導的 vicinal 分佈對 $G$ 不變，並使經典穩定性係數減少一個與群軌道直徑相關的項。
* **命題 C（Fisher–Rao 相容）**：在可積分的統計模型族上，若以 Fisher–Rao 度量定義測地線插值，則 SEKI-VRM 的梯度方向與自然梯度方向在一階近似下對齊。

> 證明材料：VRM 風險分解、比較幾何（Jacobi 場/Toponogov 型不等式框架）、Afsari 的 Karcher 收斂與唯一性條件、Fisher–Rao 測地線與距離的已知閉式/近似結果。([papers.neurips.cc][1], [vision.jhu.edu][10], [arXiv][12])

---

## 實作備忘（你可以直接開工）

* 幾何層：實作 `exp_x(v), log_x(y)`，提供 **SPD(Log-Euclidean)**、**球面/雙曲**、\*\*Fisher–Rao（特定分佈族）\*\*介面。([PubMed][4], [PMC][14])
* 等變層：提供 $G=$ 旋轉/反射/平移 的作用與軌道代表（或 quotient 投影），並做「先對齊、後插值」。([Proceedings of Machine Learning Research][9])
* 資料/評測：`make_moons/circles/swiss_roll/make_spd_matrix` + Mixup/ManifoldMixup/R-Mixup/G-CNN 基線。([scikit-learn.org][15], [arXiv][5], [Proceedings of Machine Learning Research][6])

---

如果你要，我可以把以上內容展開成**論文大綱（含章節與符號表）**與**實驗代碼雛形**（PyTorch + 幾何工具函式），並附上可直接跑在 `sklearn` toy datasets 的範例腳本。

[1]: https://papers.neurips.cc/paper/1876-vicinal-risk-minimization.pdf?utm_source=chatgpt.com "Vicinal Risk Minimization"
[2]: https://www.scholarpedia.org/article/Fisher-Rao_metric?utm_source=chatgpt.com "Fisher-Rao metric"
[3]: https://content.e-bookshelf.de/media/reading/L-7505366-733c6681c3.pdf?utm_source=chatgpt.com "Shun-ichi Amari - Information Geometry and Its Applications"
[4]: https://pubmed.ncbi.nlm.nih.gov/16788917/?utm_source=chatgpt.com "Log-Euclidean metrics for fast and simple calculus on ..."
[5]: https://arxiv.org/pdf/1710.09412?utm_source=chatgpt.com "mixup: BEYOND EMPIRICAL RISK MINIMIZATION"
[6]: https://proceedings.mlr.press/v97/verma19a/verma19a.pdf?utm_source=chatgpt.com "Manifold Mixup: Better Representations by Interpolating ..."
[7]: https://dl.acm.org/doi/abs/10.1162/089976698300017746?utm_source=chatgpt.com "Natural gradient works efficiently in learning"
[8]: https://www.jmlr.org/papers/volume21/17-678/17-678.pdf?utm_source=chatgpt.com "New Insights and Perspectives on the Natural Gradient ..."
[9]: https://proceedings.mlr.press/v48/cohenc16.html?utm_source=chatgpt.com "Group Equivariant Convolutional Networks"
[10]: https://www.vision.jhu.edu/assets/AfsariSJCO12.pdf?utm_source=chatgpt.com "on the convergence of gradient descent for finding the ..."
[11]: https://www.jstor.org/stable/41059320?utm_source=chatgpt.com "RIEMANNIAN L p CENTER OF MASS: EXISTENCE ..."
[12]: https://arxiv.org/pdf/2304.14885?utm_source=chatgpt.com "On Closed-Form Expressions for the Fisher–Rao Distance"
[13]: https://arxiv.org/abs/2306.02532?utm_source=chatgpt.com "R-Mixup: Riemannian Mixup for Biological Networks"
[14]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7516881/?utm_source=chatgpt.com "The Fisher–Rao Distance between Multivariate Normal ..."
[15]: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html?utm_source=chatgpt.com "make_moons"
[16]: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_spd_matrix.html?utm_source=chatgpt.com "make_spd_matrix"