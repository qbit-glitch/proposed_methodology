I'll provide comprehensive mathematical explanations and equations for each algorithm.

---

## **MATHEMATICAL FOUNDATIONS**

### **Notation**
- $\mathbb{R}^{m \times n}$: Real-valued matrix of dimension $m \times n$
- $\odot$: Element-wise multiplication (Hadamard product)
- $\circ$: Function composition
- $\|\cdot\|_2$: L2 norm
- $\sigma(\cdot)$: Sigmoid function
- $\text{softmax}(\cdot)$: Softmax function
- $\text{ReLU}(x) = \max(0, x)$

---

## **ALGORITHM 1: CNN-BACKBONE (Feature Extraction)**

### **Multi-Scale Feature Pyramid**

Given input image $I \in \mathbb{R}^{H \times W \times 3}$, ResNet50 produces features at multiple scales:

$$C_3 = f_3(I) \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{8} \times 512}$$

$$C_4 = f_4(C_3) \in \mathbb{R}^{\frac{H}{16} \times \frac{W}{16} \times 1024}$$

$$C_5 = f_5(C_4) \in \mathbb{R}^{\frac{H}{32} \times \frac{W}{32} \times 2048}$$

Where $f_3, f_4, f_5$ are residual blocks defined as:

$$f_\ell(x) = x + \mathcal{F}_\ell(x)$$

$$\mathcal{F}_\ell(x) = W_{\ell,2} \cdot \text{ReLU}(W_{\ell,1} \cdot x + b_{\ell,1}) + b_{\ell,2}$$

### **Flatten and Concatenation**

Reshape each feature map to sequence format:

$$C_3^{flat} = \text{Reshape}(C_3) \in \mathbb{R}^{N_3 \times 512}, \quad N_3 = \frac{HW}{64}$$

$$C_4^{flat} = \text{Reshape}(C_4) \in \mathbb{R}^{N_4 \times 1024}, \quad N_4 = \frac{HW}{256}$$

$$C_5^{flat} = \text{Reshape}(C_5) \in \mathbb{R}^{N_5 \times 2048}, \quad N_5 = \frac{HW}{1024}$$

Project to common dimension $D = 256$:

$$\tilde{C}_3 = C_3^{flat} W_3 \in \mathbb{R}^{N_3 \times 256}, \quad W_3 \in \mathbb{R}^{512 \times 256}$$

$$\tilde{C}_4 = C_4^{flat} W_4 \in \mathbb{R}^{N_4 \times 256}, \quad W_4 \in \mathbb{R}^{1024 \times 256}$$

$$\tilde{C}_5 = C_5^{flat} W_5 \in \mathbb{R}^{N_5 \times 256}, \quad W_5 \in \mathbb{R}^{2048 \times 256}$$

Concatenate:

$$\mathcal{F}_{flat} = [\tilde{C}_3; \tilde{C}_4; \tilde{C}_5] \in \mathbb{R}^{N \times 256}, \quad N = N_3 + N_4 + N_5$$

### **Positional Encoding**

Add sinusoidal positional encodings to preserve spatial information:

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/D}}\right)$$

$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/D}}\right)$$

Where $pos$ is the spatial position and $i$ is the dimension index.

$$\mathcal{F}_{pos} = \mathcal{F}_{flat} + PE \in \mathbb{R}^{N \times 256}$$

---

## **ALGORITHM 2: TRANSFORMER-ENCODER**

### **Multi-Head Self-Attention (MHSA)**

For each layer $\ell \in \{1, \ldots, 6\}$:

**Step 1: Linear Projections**

For $h = 8$ attention heads, $d_k = D/h = 32$:

$$Q^{(\ell)} = \mathcal{M}^{(\ell-1)} W_Q^{(\ell)} \in \mathbb{R}^{N \times D}$$

$$K^{(\ell)} = \mathcal{M}^{(\ell-1)} W_K^{(\ell)} \in \mathbb{R}^{N \times D}$$

$$V^{(\ell)} = \mathcal{M}^{(\ell-1)} W_V^{(\ell)} \in \mathbb{R}^{N \times D}$$

Where $W_Q^{(\ell)}, W_K^{(\ell)}, W_V^{(\ell)} \in \mathbb{R}^{D \times D}$

**Step 2: Split into Multiple Heads**

$$Q^{(\ell)} = [Q_1^{(\ell)}, Q_2^{(\ell)}, \ldots, Q_h^{(\ell)}], \quad Q_i^{(\ell)} \in \mathbb{R}^{N \times d_k}$$

Similarly for $K^{(\ell)}$ and $V^{(\ell)}$.

**Step 3: Scaled Dot-Product Attention**

For each head $i$:

$$\text{Attention}(Q_i^{(\ell)}, K_i^{(\ell)}, V_i^{(\ell)}) = \text{softmax}\left(\frac{Q_i^{(\ell)} (K_i^{(\ell)})^T}{\sqrt{d_k}}\right) V_i^{(\ell)}$$

The attention weights matrix:

$$A_i^{(\ell)} = \text{softmax}\left(\frac{Q_i^{(\ell)} (K_i^{(\ell)})^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{N \times N}$$

Where softmax is applied row-wise:

$$A_{i,jk}^{(\ell)} = \frac{\exp\left(\frac{q_{ij}^{(\ell)} \cdot k_{ik}^{(\ell)}}{\sqrt{d_k}}\right)}{\sum_{m=1}^{N} \exp\left(\frac{q_{ij}^{(\ell)} \cdot k_{im}^{(\ell)}}{\sqrt{d_k}}\right)}$$

**Step 4: Concatenate Heads**

$$\text{head}_i^{(\ell)} = A_i^{(\ell)} V_i^{(\ell)} \in \mathbb{R}^{N \times d_k}$$

$$\text{MHSA}^{(\ell)} = \text{Concat}(\text{head}_1^{(\ell)}, \ldots, \text{head}_h^{(\ell)}) W_O^{(\ell)}$$

Where $W_O^{(\ell)} \in \mathbb{R}^{D \times D}$.

**Step 5: Residual Connection and Layer Normalization**

$$\mathcal{M}'^{(\ell)} = \text{LayerNorm}(\mathcal{M}^{(\ell-1)} + \text{MHSA}^{(\ell)})$$

Layer normalization:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- $\mu = \frac{1}{D}\sum_{i=1}^D x_i$ (mean)
- $\sigma^2 = \frac{1}{D}\sum_{i=1}^D (x_i - \mu)^2$ (variance)
- $\gamma, \beta \in \mathbb{R}^D$ are learnable parameters
- $\epsilon = 10^{-6}$ for numerical stability

### **Feed-Forward Network (FFN)**

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2$$

Where:
- $W_1 \in \mathbb{R}^{D \times D_{ff}}$, typically $D_{ff} = 4D = 1024$
- $W_2 \in \mathbb{R}^{D_{ff} \times D}$
- $b_1 \in \mathbb{R}^{D_{ff}}$, $b_2 \in \mathbb{R}^D$

**Residual Connection:**

$$\mathcal{M}^{(\ell)} = \text{LayerNorm}(\mathcal{M}'^{(\ell)} + \text{FFN}(\mathcal{M}'^{(\ell)}))$$

### **Final Encoder Output**

After $L = 6$ layers:

$$\mathcal{M} = \mathcal{M}^{(6)} \in \mathbb{R}^{N \times D}$$

This is the **memory** used by all decoders.

---

## **ALGORITHM 3: DETECTION-DECODER**

### **Learnable Query Initialization**

$$Q_{det} \in \mathbb{R}^{100 \times D}$$

These are randomly initialized and learned during training.

### **Decoder Layer $\ell$**

**Step 1: Self-Attention**

Same as encoder self-attention but queries attend to themselves:

$$Q'^{(\ell)} = \text{MHSA}(Q^{(\ell-1)}, Q^{(\ell-1)}, Q^{(\ell-1)})$$

$$Q^{(\ell)}_1 = \text{LayerNorm}(Q^{(\ell-1)} + Q'^{(\ell)})$$

**Step 2: Cross-Attention with Encoder Memory**

$$Q'^{(\ell)} = \text{MHCA}(Q^{(\ell)}_1, \mathcal{M}, \mathcal{M})$$

Where Multi-Head Cross-Attention (MHCA) is defined as:

$$\text{MHCA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

$$\text{head}_i = \text{softmax}\left(\frac{(QW_i^Q)(KW_i^K)^T}{\sqrt{d_k}}\right) (VW_i^V)$$

Key difference: $Q$ comes from decoder queries, $K, V$ come from encoder memory.

$$Q^{(\ell)}_2 = \text{LayerNorm}(Q^{(\ell)}_1 + Q'^{(\ell)})$$

**Step 3: Feed-Forward Network**

$$Q'^{(\ell)} = \text{FFN}(Q^{(\ell)}_2)$$

$$Q^{(\ell)} = \text{LayerNorm}(Q^{(\ell)}_2 + Q'^{(\ell)})$$

### **Output Heads**

After 6 decoder layers, $Q^{(6)} = \mathcal{D}_{feat} \in \mathbb{R}^{100 \times D}$

For each query $i \in \{1, \ldots, 100\}$:

**Bounding Box Regression:**

$$bbox_i = \sigma(W_{bbox} \mathcal{D}_{feat}[i] + b_{bbox}) \in \mathbb{R}^4$$

Where $\sigma$ is sigmoid to normalize to $[0, 1]$.

Output format: $(x_{center}, y_{center}, width, height)$ normalized by image dimensions.

**Classification Head:**

$$p_{class,i} = \text{softmax}(W_{class} \mathcal{D}_{feat}[i] + b_{class}) \in \mathbb{R}^7$$

Classes: $\{car, scooty, bike, bus, truck, auto, background\}$

$$W_{class} \in \mathbb{R}^{D \times 7}, \quad b_{class} \in \mathbb{R}^7$$

**Color Prediction:**

$$p_{color,i} = \text{softmax}(W_{color} \mathcal{D}_{feat}[i] + b_{color}) \in \mathbb{R}^{n_{colors}}$$

**Type Prediction:**

$$p_{type,i} = \text{softmax}(W_{type} \mathcal{D}_{feat}[i] + b_{type}) \in \mathbb{R}^{n_{types}}$$

---

## **ALGORITHM 4: SEGMENTATION-DECODER**

### **Inter-Decoder Cross-Attention**

After self-attention and cross-attention with memory, add cross-attention with detection features:

**Step 3: Cross-Attention with Detection Features**

$$Q'^{(\ell)} = \text{MHCA}(Q^{(\ell)}_2, \mathcal{D}_{feat}, \mathcal{D}_{feat})$$

Explicitly:

$$\text{head}_i = \text{softmax}\left(\frac{(Q^{(\ell)}_2 W_i^Q)(\mathcal{D}_{feat} W_i^K)^T}{\sqrt{d_k}}\right) (\mathcal{D}_{feat} W_i^V)$$

This allows segmentation queries to "see" where vehicles are located.

$$Q^{(\ell)}_3 = \text{LayerNorm}(Q^{(\ell)}_2 + Q'^{(\ell)})$$

**Step 4: Feed-Forward**

$$Q^{(\ell)} = \text{LayerNorm}(Q^{(\ell)}_3 + \text{FFN}(Q^{(\ell)}_3))$$

### **Mask Generation Head**

Final queries: $\mathcal{S}_{feat} = Q^{(6)} \in \mathbb{R}^{50 \times D}$

**Pixel Decoder:**

Upsample encoder features back to image resolution using FPN-style architecture:

$$F_{mask} = \text{FPN}(\mathcal{M}, C_3, C_4, C_5) \in \mathbb{R}^{\frac{H}{4} \times \frac{W}{4} \times D_{mask}}$$

**Per-Query Mask Prediction:**

For each segmentation query $i$:

$$mask_i = \text{Upsample}(\mathcal{S}_{feat}[i]^T F_{mask}) \in \mathbb{R}^{H \times W}$$

$$mask_i = \sigma(W_{up} \star (q_i^T \cdot F_{mask}))$$

Where $\star$ denotes transposed convolution for upsampling.

**Final Segmentation Masks:**

$$mask_{driveway} = \max_{i \in I_{driveway}} mask_i$$

$$mask_{footpath} = \max_{j \in I_{footpath}} mask_j$$

Where $I_{driveway}$ and $I_{footpath}$ are query indices assigned to each class.

**Alternative (with classification):**

$$p_{seg,i} = \text{softmax}(W_{seg} \mathcal{S}_{feat}[i] + b_{seg}) \in \mathbb{R}^3$$

Classes: $\{driveway, footpath, background\}$

Final mask:

$$M[h,w] = \arg\max_c \sum_{i: class_i = c} mask_i[h,w] \cdot p_{seg,i}[c]$$

---

## **ALGORITHM 5: PLATE-DECODER**

### **Constrained Attention via Detection Features**

The key innovation: plate queries attend to detection features to constrain search space.

**Cross-Attention Computation:**

$$\text{head}_i = \text{softmax}\left(\frac{(Q_{plate} W_i^Q)(\mathcal{D}_{feat} W_i^K)^T}{\sqrt{d_k}}\right) (\mathcal{D}_{feat} W_i^V)$$

Attention weight matrix:

$$A_{ij} = \frac{\exp\left(\frac{q_{plate,i} \cdot k_{det,j}}{\sqrt{d_k}}\right)}{\sum_{k=1}^{100} \exp\left(\frac{q_{plate,i} \cdot k_{det,k}}{\sqrt{d_k}}\right)}$$

**Interpretation:**
- If $A_{ij}$ is high, plate query $i$ focuses on detection $j$
- Plate decoder learns: "Search for plates only near detected vehicles"
- Reduces false positives on billboards, signs, etc.

### **Spatial Guidance**

Detection features encode spatial location. The cross-attention implicitly computes:

$$\mathcal{P}_{feat}[i] \approx \sum_{j=1}^{100} A_{ij} \cdot \mathcal{D}_{feat}[j]$$

Where $A_{ij}$ is high when:
1. Detection $j$ is a vehicle
2. Spatial proximity suggests plate location
3. Vehicle type matches expected plate position (e.g., rear for cars)

### **Plate Bounding Box Head**

$$plate\_bbox_i = \sigma(W_{plate} \mathcal{P}_{feat}[i] + b_{plate}) \in \mathbb{R}^4$$

Output: $(x_{center}, y_{center}, width, height)$ in normalized coordinates.

---

## **ALGORITHM 6: OCR-DECODER**

### **Critical Cross-Attention: Plate → OCR**

$$\text{head}_i = \text{softmax}\left(\frac{(Q_{ocr} W_i^Q)(\mathcal{P}_{feat} W_i^K)^T}{\sqrt{d_k}}\right) (\mathcal{P}_{feat} W_i^V)$$

**Spatial Cropping via Attention:**

The attention mechanism effectively performs ROI pooling:

$$\mathcal{O}_{feat}[i] = \sum_{j=1}^{50} A_{ij} \cdot \mathcal{P}_{feat}[j]$$

Where $A_{ij}$ peaks when:
- Plate $j$ is valid (high detection confidence)
- OCR character position $i$ aligns with plate $j$'s spatial extent

### **CTC (Connectionist Temporal Classification)**

OCR output is a sequence prediction problem. Use CTC loss for variable-length text.

**Character Logits:**

$$z_i^{(t)} = W_{char} \mathcal{O}_{feat}[t] + b_{char} \in \mathbb{R}^{|\mathcal{A}| + 1}$$

Where:
- $t \in \{1, \ldots, 20\}$ is the time step (character position)
- $\mathcal{A}$ is the alphabet (A-Z, 0-9, special chars)
- Additional blank token for CTC

**Character Probabilities:**

$$p(a | t) = \frac{\exp(z_i^{(t)}[a])}{\sum_{a' \in \mathcal{A} \cup \{blank\}} \exp(z_i^{(t)}[a'])}$$

**CTC Alignment:**

A valid alignment $\pi$ is a sequence $\pi = (\pi_1, \ldots, \pi_T)$ where $\pi_t \in \mathcal{A} \cup \{blank\}$.

**Collapsing Function:**

Remove repeated characters and blanks:

$$\mathcal{B}(\pi) = \text{RemoveRepeats}(\text{RemoveBlanks}(\pi))$$

Example: $\mathcal{B}([D, D, blank, L, 0, 1]) = "DL01"$

**CTC Probability:**

$$p(y | \mathcal{O}_{feat}) = \sum_{\pi: \mathcal{B}(\pi) = y} \prod_{t=1}^T p(\pi_t | t)$$

**Decoding (Inference):**

Best path decoding:

$$\hat{y} = \mathcal{B}\left(\arg\max_{\pi} \prod_{t=1}^T p(\pi_t | t)\right)$$

Or beam search for better accuracy.

---

## **ALGORITHM 7: TRACKING-DECODER**

### **Temporal Query Initialization**

Augment tracking queries with motion priors from previous frame:

$$Q_{track}^{(0)} = [q_1, q_2, \ldots, q_m] \in \mathbb{R}^{m \times D}$$

For existing tracks from $T_{prev}$:

$$q_i = \text{MLP}([f_{appear,i}; \Delta x_i; \Delta y_i; v_i]) \in \mathbb{R}^D$$

Where:
- $f_{appear,i}$: Appearance feature from previous frame
- $\Delta x_i, \Delta y_i$: Position change (velocity)
- $v_i$: Speed magnitude

### **Dual Cross-Attention**

**Cross-Attention #1: Detection Features**

$$Q'^{(\ell)}_3 = \text{MHCA}(Q^{(\ell)}_2, \mathcal{D}_{feat}, \mathcal{D}_{feat})$$

This provides:
- Current vehicle locations
- Vehicle classes for association
- Appearance features

**Cross-Attention #2: Segmentation Features**

$$Q'^{(\ell)}_4 = \text{MHCA}(Q^{(\ell)}_3, \mathcal{S}_{feat}, \mathcal{S}_{feat})$$

This provides:
- Spatial context (driveway/footpath)
- Overlap information
- Scene understanding

**Combined Update:**

$$Q^{(\ell)}_4 = \text{LayerNorm}(Q^{(\ell)}_3 + Q'^{(\ell)}_4)$$

### **Track Association**

**Similarity Matrix:**

Compute pairwise similarity between tracking queries and detections:

$$S_{ij} = \frac{Q_{track}[i]^T \mathcal{D}_{feat}[j]}{\|Q_{track}[i]\|_2 \|\mathcal{D}_{feat}[j]\|_2}$$

**Learned Association:**

$$A_{ij} = \text{MLP}([S_{ij}; \text{IoU}(bbox_{track,i}, bbox_{det,j}); \Delta t_{ij}])$$

Where:
- $\text{IoU}(\cdot)$: Intersection over Union between predicted and detected boxes
- $\Delta t_{ij}$: Time difference

**Hungarian Algorithm:**

Find optimal one-to-one assignment:

$$\text{matches} = \arg\max_{\pi} \sum_{i} A_{i,\pi(i)}$$

Subject to: each track matches at most one detection.

**Implementation:**

$$\text{cost\_matrix}[i,j] = -A_{ij}$$

$$\text{matches} = \text{Hungarian}(\text{cost\_matrix})$$

### **Trajectory Prediction**

For each matched track $i$:

$$\text{trajectory}_i^{(t)} = \text{Kalman-Update}(\text{trajectory}_i^{(t-1)}, bbox_{det,\pi(i)})$$

Or learned prediction:

$$[\Delta x, \Delta y, \Delta w, \Delta h] = \text{MLP}(\mathcal{T}_{feat}[i])$$

$$bbox_i^{(t+1)} = bbox_i^{(t)} + [\Delta x, \Delta y, \Delta w, \Delta h]$$

### **Stopped Time Update**

**Velocity Threshold:**

$$v_i = \sqrt{(\Delta x_i)^2 + (\Delta y_i)^2}$$

$$\text{is\_stopped}_i = \begin{cases} 
1 & \text{if } v_i < \epsilon_{vel} \\
0 & \text{otherwise}
\end{cases}$$

Where $\epsilon_{vel} = 0.5$ pixels/frame (tunable).

**Time Accumulation:**

$$stopped\_time_i^{(t)} = \begin{cases}
stopped\_time_i^{(t-1)} + \Delta t & \text{if } \text{is\_stopped}_i = 1 \\
0 & \text{otherwise}
\end{cases}$$

Where $\Delta t$ is the frame interval (e.g., 1/30 seconds for 30 FPS).

---

## **ALGORITHM 8: GENERATE-ALERTS**

### **Intersection over Union (IoU)**

For bounding box $bbox = (x, y, w, h)$ and segmentation mask $M \in \{0, 1\}^{H \times W}$:

**Convert bbox to mask:**

$$M_{bbox}[i,j] = \begin{cases}
1 & \text{if } x \leq j < x+w \text{ and } y \leq i < y+h \\
0 & \text{otherwise}
\end{cases}$$

**Compute IoU:**

$$\text{Intersection} = \sum_{i,j} M_{bbox}[i,j] \cdot M_{driveway}[i,j]$$

$$\text{Union} = \sum_{i,j} \max(M_{bbox}[i,j], M_{driveway}[i,j])$$

$$\text{IoU} = \frac{\text{Intersection}}{\text{Union}}$$

**Alternative (continuous):**

If segmentation outputs probabilities $P_{driveway}[i,j] \in [0,1]$:

$$\text{overlap} = \frac{\sum_{i,j} M_{bbox}[i,j] \cdot P_{driveway}[i,j]}{\sum_{i,j} M_{bbox}[i,j]}$$

This represents the fraction of the bounding box overlapping with driveway.

### **Alert Condition**

$$\text{alert}_i = \begin{cases}
\text{RED} & \text{if } (stopped\_time_i > 120) \land (\text{overlap}_i > 0.3) \\
\text{NORMAL} & \text{otherwise}
\end{cases}$$

Logical form:

$$\text{alert}_i = \mathbb{I}[stopped\_time_i > \tau_{time}] \cdot \mathbb{I}[\text{overlap}_i > \tau_{overlap}]$$

Where $\mathbb{I}[\cdot]$ is the indicator function, $\tau_{time} = 120$ seconds, $\tau_{overlap} = 0.3$.

---

## **ALGORITHM 9: MULTI-HEAD-CROSS-ATTENTION (Detailed)**

### **Input Dimensions**

- Queries: $Q \in \mathbb{R}^{n_q \times D}$
- Keys: $K \in \mathbb{R}^{n_k \times D}$
- Values: $V \in \mathbb{R}^{n_k \times D}$
- Number of heads: $h = 8$
- Dimension per head: $d_k = D/h = 32$

### **Linear Projections**

For head $i \in \{1, \ldots, h\}$:

$$Q_i = Q W_i^Q \in \mathbb{R}^{n_q \times d_k}, \quad W_i^Q \in \mathbb{R}^{D \times d_k}$$

$$K_i = K W_i^K \in \mathbb{R}^{n_k \times d_k}, \quad W_i^K \in \mathbb{R}^{D \times d_k}$$

$$V_i = V W_i^V \in \mathbb{R}^{n_k \times d_k}, \quad W_i^V \in \mathbb{R}^{D \times d_k}$$

### **Attention Scores**

$$S_i = \frac{Q_i K_i^T}{\sqrt{d_k}} \in \mathbb{R}^{n_q \times n_k}$$

Element-wise:

$$S_i[m,n] = \frac{1}{\sqrt{d_k}} \sum_{j=1}^{d_k} Q_i[m,j] \cdot K_i[n,j]$$

### **Attention Weights**

Apply softmax row-wise:

$$A_i[m,n] = \frac{\exp(S_i[m,n])}{\sum_{n'=1}^{n_k} \exp(S_i[m,n'])}$$

Properties:
- $\sum_{n=1}^{n_k} A_i[m,n] = 1$ for all $m$
- $A_i[m,n] \in [0, 1]$

### **Weighted Sum**

$$\text{head}_i = A_i V_i \in \mathbb{R}^{n_q \times d_k}$$

Element-wise:

$$\text{head}_i[m,j] = \sum_{n=1}^{n_k} A_i[m,n] \cdot V_i[n,j]$$

### **Concatenation and Output Projection**

$$O_{concat} = [\text{head}_1, \text{head}_2, \ldots, \text{head}_h] \in \mathbb{R}^{n_q \times D}$$

$$O = O_{concat} W_O + b_O \in \mathbb{R}^{n_q \times D}$$

Where $W_O \in \mathbb{R}^{D \times D}$, $b_O \in \mathbb{R}^D$.

### **Computational Complexity**

- Attention matrix: $\mathcal{O}(n_q \cdot n_k \cdot D)$
- Softmax: $\mathcal{O}(n_q \cdot n_k)$
- Weighted sum: $\mathcal{O}(n_q \cdot n_k \cdot D)$
- **Total:** $\mathcal{O}(n_q \cdot n_k \cdot D)$

For self-attention ($n_q = n_k = N$): $\mathcal{O}(N^2 \cdot D)$ — quadratic in sequence length.

---

## **ALGORITHM 10: TRAINING PROCEDURE**

### **Loss Functions**

#### **1. Detection Loss**

**Classification Loss (Focal Loss):**

$$\mathcal{L}_{cls} = -\frac{1}{N_{pos}} \sum_{i=1}^{100} \alpha_t (1 - p_i)^\gamma \log(p_i)$$

Where:
- $p_i = p_{class,i}[y_i]$ is the predicted probability for ground truth class
- $\alpha_t \in [0, 1]$ balances positive/negative examples
- $\gamma \geq 0$ focuses on hard examples ($\gamma = 2$ typical)
- $N_{pos}$ is the number of positive (matched) queries

**Bounding Box Loss (GIoU):**

$$\mathcal{L}_{bbox} = \frac{1}{N_{pos}} \sum_{i \in \text{matched}} [\lambda_1 \mathcal{L}_1(bbox_i, \hat{bbox}_i) + \lambda_2 \mathcal{L}_{GIoU}(bbox_i, \hat{bbox}_i)]$$

L1 loss:

$$\mathcal{L}_1(bbox, \hat{bbox}) = |x - \hat{x}| + |y - \hat{y}| + |w - \hat{w}| + |h - \hat{h}|$$

Generalized IoU loss:

$$\mathcal{L}_{GIoU}(B_1, B_2) = 1 - \text{IoU}(B_1, B_2) + \frac{|C \setminus (B_1 \cup B_2)|}{|C|}$$

Where $C$ is the smallest box enclosing both $B_1$ and $B_2$.

**Total Detection Loss:**

$$\mathcal{L}_{det} = \mathcal{L}_{cls} + \mathcal{L}_{bbox} + \mathcal{L}_{color} + \mathcal{L}_{type}$$

#### **2. Segmentation Loss**

**Cross-Entropy Loss:**

$$\mathcal{L}_{CE} = -\frac{1}{HW} \sum_{h=1}^H \sum_{w=1}^W \sum_{c \in \{D,F,B\}} y_{hw}^c \log(p_{hw}^c)$$

Where:
- $y_{hw}^c \in \{0, 1\}$ is ground truth (one-hot)
- $p_{hw}^c$ is predicted probability for class $c$ (Driveway, Footpath, Background)

**Dice Loss:**

$$\mathcal{L}_{Dice} = 1 - \frac{2 \sum_{h,w} p_{hw}^c y_{hw}^c}{\sum_{h,w} p_{hw}^c + \sum_{h,w} y_{hw}^c + \epsilon}$$

**Total Segmentation Loss:**

$$\mathcal{L}_{seg} = \mathcal{L}_{CE} + \mathcal{L}_{Dice}$$

#### **3. Plate Detection Loss**

Same structure as detection loss:

$$\mathcal{L}_{plate} = \mathcal{L}_{cls}^{plate} + \mathcal{L}_{bbox}^{plate}$$

#### **4. OCR Loss (CTC Loss)**

$$\mathcal{L}_{CTC} = -\log p(y | \mathcal{O}_{feat}) = -\log \sum_{\pi: \mathcal{B}(\pi) = y} \prod_{t=1}^T p(\pi_t | t)$$

Computed efficiently using forward-backward algorithm:

**Forward Variable:**

$$\alpha_t(s) = p(\pi_{1:t} \text{ ends at state } s)$$

Recursion:

$$\alpha_t(s) = (\alpha_{t-1}(s) + \alpha_{t-1}(s-1)) \cdot p(l_s | t)$$

Where $l_s$ is the label at state $s$.

**Backward Variable:**

$$\beta_t(s) = p(\pi_{t+1:T} \text{ starts at state } s)$$

**Total Probability:**

$$p(y | \mathcal{O}_{feat}) = \sum_s \alpha_T(s)$$

#### **5. Tracking Loss**

**Association Loss (Cross-Entropy):**

$$\mathcal{L}_{assoc} = -\sum_{i=1}^{m} \sum_{j=1}^{n} y_{ij} \log(A_{ij})$$

Where $y_{ij} \in \{0, 1\}$ indicates if track $i$ matches detection $j$.

**Trajectory Loss (Smooth L1):**

$$\mathcal{L}_{traj} = \frac{1}{m} \sum_{i=1}^m \text{SmoothL1}(bbox_i^{pred}, bbox_i^{gt})$$

$$\text{SmoothL1}(x) = \begin{cases}
0.5x^2 & \text{if } |x| < 1 \\
|x| - 0.5 & \text{otherwise}
\end{cases}$$

**Total Tracking Loss:**

$$\mathcal{L}_{track} = \mathcal{L}_{assoc} + \mathcal{L}_{traj}$$

### **Multi-Task Loss**

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{det} + \lambda_2 \mathcal{L}_{seg} + \lambda_3 \mathcal{L}_{plate} + \lambda_4 \mathcal{L}_{ocr} + \lambda_5 \mathcal{L}_{track}$$

**Weight Selection:**

Option 1 (Fixed weights):
- $\lambda_1 = 1.0, \lambda_2 = 1.0, \lambda_3 = 2.0, \lambda_4 = 1.5, \lambda_5 = 1.0$

Option 2 (Uncertainty weighting):

$$\mathcal{L}_{total} = \sum_{i=1}^5 \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log(\sigma_i)$$

Where $\sigma_i$ are learnable parameters representing task uncertainty.

### **Gradient Computation**

**Backpropagation:**

$$\frac{\partial \mathcal{L}_{total}}{\partial \Theta} = \sum_{i=1}^5 \lambda_i \frac{\partial \mathcal{L}_i}{\partial \Theta}$$

For inter-decoder connections (e.g., Plate → OCR):

$$\frac{\partial \mathcal{L}_{ocr}}{\partial W_{plate}} = \frac{\partial \mathcal{L}_{ocr}}{\partial \mathcal{O}_{feat}} \cdot \frac{\partial \mathcal{O}_{feat}}{\partial \mathcal{P}_{feat}} \cdot \frac{\partial \mathcal{P}_{feat}}{\partial W_{plate}}$$

This shows gradients flow through the entire chain.

### **Optimizer (Adam)**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\Theta_{t+1} = \Theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Where:
- $g_t = \nabla_\Theta \mathcal{L}_{total}$ is the gradient
- $\beta_1 = 0.9, \beta_2 = 0.999$ (momentum parameters)
- $\eta = 10^{-4}$ (learning rate)
- $\epsilon = 10^{-8}$ (numerical stability)

### **Learning Rate Schedule**

Warmup + Cosine decay:

$$\eta_t = \begin{cases}
\eta_{base} \cdot \frac{t}{T_{warmup}} & \text{if } t \leq T_{warmup} \\
\eta_{min} + \frac{1}{2}(\eta_{base} - \eta_{min})\left(1 + \cos\left(\frac{t - T_{warmup}}{T_{max} - T_{warmup}}\pi\right)\right) & \text{otherwise}
\end{cases}$$

Typical values:
- $\eta_{base} = 10^{-4}$
- $\eta_{min} = 10^{-6}$
- $T_{warmup} = 1000$ iterations
- $T_{max} = $ total training iterations

---

## **COMPLEXITY ANALYSIS**

### **Time Complexity per Forward Pass**

1. **CNN Backbone:** $\mathcal{O}(HW \cdot C_{in} \cdot C_{out})$ ≈ $\mathcal{O}(HW)$ for fixed channels

2. **Transformer Encoder (6 layers):**
   - Self-attention: $\mathcal{O}(N^2 D)$ per layer
   - FFN: $\mathcal{O}(ND^2)$ per layer
   - **Total:** $\mathcal{O}(6(N^2 D + ND^2))$

3. **Decoders (5 decoders, 6 layers each):**
   - Detection: $\mathcal{O}(100 \cdot N \cdot D)$ (cross-attention with memory)
   - Segmentation: $\mathcal{O}(50 \cdot N \cdot D + 50 \cdot 100 \cdot D)$ (memory + detection)
   - Plate: $\mathcal{O}(50 \cdot N \cdot D + 50 \cdot 100 \cdot D)$
   - OCR: $\mathcal{O}(20 \cdot N \cdot D + 20 \cdot 50 \cdot D)$
   - Tracking: $\mathcal{O}(m \cdot N \cdot D + m \cdot 100 \cdot D + m \cdot 50 \cdot D)$

4. **Output Heads:** $\mathcal{O}(n_{queries} \cdot D)$ for each task

**Dominant Term:** Encoder self-attention $\mathcal{O}(N^2 D)$ where $N = N_3 + N_4 + N_5 \approx \frac{HW}{64} + \frac{HW}{256} + \frac{HW}{1024} \approx \frac{21HW}{1024}$

For $512 \times 512$ images: $N \approx 5461$, so encoder attention is $\mathcal{O}(29M \cdot D)$ operations.

### **Space Complexity**

- Encoder memory: $\mathcal{O}(N \cdot D)$
- All decoder queries: $\mathcal{O}((100 + 50 + 50 + 20 + m) \cdot D)$
- Attention weight matrices: $\mathcal{O}(N^2)$ per layer (can be checkpointed)

**Total:** $\mathcal{O}(N^2 + ND)$ ≈ $\mathcal{O}(N^2)$ for large $N$.

---

## **KEY MATHEMATICAL INSIGHTS**

1. **Cross-Attention as Soft Selection:**
   $$\mathcal{O}_{feat}[i] = \sum_{j} A_{ij} \mathcal{P}_{feat}[j]$$
   OCR query $i$ performs a weighted average over plate features, with weights determined by learned relevance.

2. **Inter-Decoder Information Flow:**
   $$\frac{\partial \mathcal{L}_{ocr}}{\partial \mathcal{P}_{feat}} \neq 0$$
   Gradients from OCR loss flow back to plate decoder, making plate detection optimize for OCR-friendly features.

3. **Alert as Logical Conjunction:**
   $$alert = \mathbb{I}[time > \tau_1] \cdot \mathbb{I}[overlap > \tau_2]$$
   Non-differentiable post-processing, not learned end-to-end.

4. **Positional Encoding Preservation:**
   Sinusoidal encoding ensures transformer maintains spatial awareness:
   $$PE(pos + k, i) \text{ relates to } PE(pos, i) \text{ via known phase shift}$$

5. **Hierarchical Attention Chains:**
   $$\text{Image} \xrightarrow{\text{Encoder}} \mathcal{M} \xrightarrow{\text{Detection}} \mathcal{D}_{feat} \xrightarrow{\text{Plate}} \mathcal{P}_{feat} \xrightarrow{\text{OCR}} \mathcal{O}$$
   Each stage refines representations for downstream tasks.

---
