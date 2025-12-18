
**You MUST use a custom dataset** for end-to-end training, but you can **leverage public datasets strategically** through transfer learning and task-specific pre-training.

---

## **DETAILED ANALYSIS**

### **Why You Need a Custom Dataset**

#### **1. Multi-Task Annotation Requirement**

Your architecture requires **simultaneous annotations** for all 5 tasks on the **same images**:

$$\mathcal{D}_{train} = \{(I_i, Y_i^{det}, Y_i^{seg}, Y_i^{plate}, Y_i^{ocr}, Y_i^{track})\}_{i=1}^N$$

For each image $I_i$, you need:
- $Y_i^{det}$: Vehicle bounding boxes + classes (car, scooty, bike, bus, truck, auto) + colors + types
- $Y_i^{seg}$: Pixel-wise driveway and footpath segmentation masks
- $Y_i^{plate}$: License plate bounding boxes
- $Y_i^{ocr}$: License plate text transcriptions
- $Y_i^{track}$: Track IDs across frames (for video)

**Problem with Public Datasets:**

No single public dataset has all these annotations together:

| Dataset | Detection | Segmentation | Plates | OCR | Tracking | Driveway/Footpath |
|---------|-----------|--------------|--------|-----|----------|-------------------|
| COCO | ✅ (generic) | ✅ (generic) | ❌ | ❌ | ❌ | ❌ |
| Cityscapes | ✅ | ✅ (road) | ❌ | ❌ | ❌ | ⚠️ (road only) |
| BDD100K | ✅ | ✅ | ❌ | ❌ | ✅ | ⚠️ (road only) |
| CCPD | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| OpenALPR | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| MOT17 | ✅ (people) | ❌ | ❌ | ❌ | ✅ | ❌ |

**Critical Gap:** None have **driveway/footpath segmentation** combined with vehicle detection and plate OCR.

#### **2. Domain-Specific Requirements**

Your use case has unique characteristics:

**Alert Logic Dependency:**
$$alert = \mathbb{I}[stopped\_time > 120s] \land \mathbb{I}[\text{IoU}(bbox, driveway) > 0.3]$$

This requires:
- Driveway segmentation specific to your deployment environment
- Stopped vehicle scenarios (not common in driving datasets)
- Overlapping violations (vehicle blocking driveway)

**Domain Shift Issues:**

Public datasets have different characteristics:

| Characteristic | Public Datasets | Your Scenario |
|----------------|----------------|---------------|
| **Camera angle** | Ego-centric (dashcam) | Fixed surveillance camera |
| **Scene type** | Highways, city streets | Residential driveways, parking areas |
| **Vehicle distribution** | Moving traffic | Stopped/parked vehicles |
| **Lighting** | Daytime, various weather | 24/7 (day/night/rain) |
| **License plates** | Country-specific format | India (e.g., "DL01AB1234") |
| **Resolution** | Variable | Fixed CCTV resolution |

**Mathematical Impact:**

The learned distribution $p_{model}(Y|X; \Theta)$ will differ significantly:

$$D_{KL}(p_{public} \| p_{target}) \text{ is large}$$

Where:
- $p_{public}$: Distribution of public dataset
- $p_{target}$: Distribution of your deployment scenario

#### **3. Class Label Mismatch**

Your specific requirements:

**Vehicle Classes:**
- Your classes: {car, scooty, bike, bus, truck, auto}
- COCO classes: {car, bus, truck, motorcycle} — missing "scooty" and "auto" (Indian-specific)
- Cityscapes: {car, truck, bus, motorcycle, bicycle} — similar mismatch

**Segmentation Classes:**
- Your classes: {driveway, footpath, background}
- Cityscapes: {road, sidewalk, building, ...} — 30+ classes, no "driveway" specifically
- COCO-Stuff: Generic categories, no specific driveway vs footpath distinction

**License Plate Format:**
- Indian format: "DL01AB1234" (2 letters, 2 digits, 2 letters, 4 digits)
- Public datasets: Mostly Chinese, European, or American formats
- OCR character distribution differs significantly

---

## **HYBRID APPROACH: Strategic Use of Public Datasets**

While you need a custom dataset for **final training**, you can leverage public datasets strategfully:

### **Strategy 1: Multi-Stage Training Pipeline**

```
Stage 1: Pre-training on Public Data (Feature Learning)
    ↓
Stage 2: Task-Specific Fine-tuning on Public Data
    ↓
Stage 3: Joint Fine-tuning on Custom Data
    ↓
Stage 4: Domain Adaptation on Custom Data
```

#### **Stage 1: Backbone Pre-training**

**Use:** ImageNet, COCO

**Goal:** Learn general visual features

$$\Theta_{backbone} \leftarrow \text{Pre-trained ResNet50/Swin Transformer}$$

**Benefit:** 
- Convergence speedup: 3-5× faster
- Better feature extraction
- Reduced custom data requirement

**Evidence from Literature:**
- Transfer learning reduces required data by 50-80% (Yosinski et al., 2014)
- Pre-trained backbones improve mAP by 5-10% (He et al., 2020)

#### **Stage 2: Task-Specific Pre-training**

**For each task separately:**

| Task | Public Dataset | Use Case |
|------|---------------|----------|
| Detection | COCO + BDD100K | Vehicle detection (generic) |
| Segmentation | Cityscapes + Mapillary | Road segmentation (adapt to driveway) |
| Plate Detection | CCPD (Chinese plates) | Plate localization (format differs but spatial patterns similar) |
| OCR | IIIT5K, SynthText | Character recognition (add Indian samples) |
| Tracking | MOT17, BDD100K | Multi-object tracking |

**Implementation:**

Train each decoder independently:

$$\Theta_{det} \leftarrow \arg\min_\Theta \mathbb{E}_{(X,Y) \sim \mathcal{D}_{COCO}} [\mathcal{L}_{det}(f_{det}(X; \Theta), Y)]$$

$$\Theta_{seg} \leftarrow \arg\min_\Theta \mathbb{E}_{(X,Y) \sim \mathcal{D}_{Cityscapes}} [\mathcal{L}_{seg}(f_{seg}(X; \Theta), Y)]$$

**Benefits:**
- Each decoder starts with reasonable initialization
- Reduced overfitting on small custom dataset
- Learn general patterns (e.g., "plates are rectangular, near vehicles")

**Limitations:**
- Tasks trained independently (no inter-decoder learning)
- Domain gap still exists

#### **Stage 3: Custom Data Collection**

**Minimum Dataset Size Estimation:**

Based on task complexity and transfer learning:

$$N_{min} = \frac{N_{params} \cdot k}{n_{epochs} \cdot \alpha_{transfer}}$$

Where:
- $N_{params}$: Number of trainable parameters
- $k$: Examples per parameter (typically 5-10 for fine-tuning)
- $n_{epochs}$: Training epochs
- $\alpha_{transfer}$: Transfer learning efficiency (0.2-0.5)

**For your architecture:**
- $N_{params} \approx 50M$ (ResNet50 + Transformer + Decoders)
- With pre-training: $N_{min} \approx \frac{50M \cdot 5}{100 \cdot 0.3} \approx 8,333$ images

**Practical Recommendation:**

| Data Type | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **Annotated Images** | 5,000 | 10,000 | 50,000+ |
| **Video Frames** | 10,000 | 30,000 | 100,000+ |
| **Unique Plates** | 500 | 2,000 | 10,000+ |
| **Violation Cases** | 1,000 | 3,000 | 10,000+ |

**Data Diversity Requirements:**

$$\text{Entropy}(\mathcal{D}) = -\sum_{c \in \mathcal{C}} p(c) \log p(c)$$

Maximize entropy across:
- Time of day: Morning, afternoon, evening, night (4 bins)
- Weather: Sunny, cloudy, rainy, foggy (4 bins)
- Scene: Residential, commercial, parking lot (3 bins)
- Vehicle types: Balanced distribution across 6 classes
- Occlusion levels: Clear, partial, heavy (3 bins)

Total scenarios: $4 \times 4 \times 3 \times 6 \times 3 = 864$ combinations

Minimum per scenario: $\frac{10,000}{864} \approx 12$ samples

#### **Stage 4: Joint Training on Custom Data**

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{det} + \lambda_2 \mathcal{L}_{seg} + \lambda_3 \mathcal{L}_{plate} + \lambda_4 \mathcal{L}_{ocr} + \lambda_5 \mathcal{L}_{track}$$

Now all tasks trained simultaneously with inter-decoder cross-attention active.

---

### **Strategy 2: Synthetic Data Augmentation**

**Generate synthetic training data:**

#### **A. Plate OCR Augmentation**

Generate synthetic license plates:

$$I_{plate}^{synthetic} = \text{Render}(text, font, background, distortion)$$

Parameters:
- Text: Sample from Indian plate format regex: `[A-Z]{2}\d{2}[A-Z]{2}\d{4}`
- Fonts: 10-15 Indian vehicle plate fonts
- Backgrounds: Colors (white, yellow for commercial)
- Distortions: Rotation (±15°), perspective, blur, noise

**Advantage:** Can generate infinite OCR training samples

**Limitation:** Synthetic-to-real domain gap

#### **B. Scene Composition**

Composite vehicles into driveway scenes:

$$I_{composite} = \alpha \cdot I_{vehicle} + (1 - \alpha) \cdot I_{background}$$

Where $\alpha$ is a blending mask.

**Tools:** 
- Copy-paste augmentation (Simple Copy-Paste, Ghiasi et al., 2021)
- GAN-based scene generation (DALL-E, Stable Diffusion)

#### **C. Physics Simulation**

Use game engines (CARLA, GTA V):

$$\mathcal{D}_{sim} = \{(I_i^{sim}, Y_i^{sim})\}$$

Automatically generate:
- Perfect bounding boxes
- Segmentation masks
- Tracking sequences

**Sim-to-Real Gap:**

Apply domain randomization:
- Randomize lighting: $L \sim \text{Uniform}(0.3, 1.0)$
- Randomize textures: Sample from texture database
- Randomize weather: Fog density, rain intensity

---

### **Strategy 3: Active Learning**

Iteratively select most informative samples:

$$x^* = \arg\max_{x \in \mathcal{D}_{unlabeled}} U(x; \Theta)$$

Where $U(x; \Theta)$ is an uncertainty measure:

**Uncertainty Measures:**

1. **Prediction Entropy:**
$$U(x) = -\sum_{c} p(c|x) \log p(c|x)$$

2. **Query-By-Committee:**
$$U(x) = \text{Variance}(\{f_1(x), f_2(x), \ldots, f_M(x)\})$$

Train $M$ models with different initializations.

3. **Expected Model Change:**
$$U(x) = \mathbb{E}_{y \sim p(y|x)} [\|\nabla_\Theta \mathcal{L}(x, y)\|]$$

**Active Learning Loop:**

```
1. Train model on initial labeled set: $\mathcal{D}_L$
2. For t = 1 to T:
    a. Score unlabeled samples: $U(x)$ for $x \in \mathcal{D}_U$
    b. Select top-K uncertain samples: $\mathcal{D}_{query}$
    c. Human annotation: Get labels for $\mathcal{D}_{query}$
    d. Update: $\mathcal{D}_L \leftarrow \mathcal{D}_L \cup \mathcal{D}_{query}$
    e. Retrain model on updated $\mathcal{D}_L$
```

**Benefit:** Reduce annotation cost by 30-50%

---

### **Strategy 4: Weak Supervision**

Use cheaper annotations:

#### **Image-Level Labels**

Instead of full bounding boxes, just: "This image contains a car"

$$Y_i^{weak} = \{car: 1, bus: 0, truck: 0, \ldots\}$$

Train with Multiple Instance Learning (MIL):

$$\mathcal{L}_{MIL} = -\sum_i \sum_c y_i^c \log\left(\max_j p_c(bbox_j | I_i)\right)$$

#### **Bounding Boxes Only (No Segmentation Masks)**

Use GrabCut or SAM (Segment Anything Model) to generate pseudo-masks:

$$M_{pseudo} = \text{SAM}(I, bbox_{vehicle})$$

Then:
$$\mathcal{L}_{seg} = \mathcal{L}_{seg}(M_{pred}, M_{pseudo}) \cdot \text{confidence}(M_{pseudo})$$

#### **Unlabeled Video Sequences**

Use self-supervised tracking:

$$\mathcal{L}_{self-track} = \|f(I_t, bbox_t) - f(I_{t+1}, bbox_{t+1})\|_2$$

Enforce feature consistency across frames.

---

## **PRACTICAL ROADMAP**

### **Phase 1: Proof of Concept (1-2 months)**

**Dataset:**
- Collect: 1,000 custom images (all tasks annotated)
- Public: Use pre-trained weights from COCO, Cityscapes, CCPD

**Training:**
- Stage 1: Use pre-trained backbone
- Stage 2: Fine-tune individual decoders on public data
- Stage 3: Joint training on 1,000 custom images

**Goal:** Validate architecture works, identify failure modes

### **Phase 2: Pilot Deployment (3-6 months)**

**Dataset:**
- Collect: 10,000 custom images
- Use active learning to prioritize annotations
- Generate 50,000 synthetic OCR samples

**Training:**
- Full multi-task training with inter-decoder attention
- Domain adaptation techniques (adversarial training)

**Goal:** Achieve acceptable performance for limited deployment

### **Phase 3: Production (6-12 months)**

**Dataset:**
- Collect: 50,000+ custom images
- Continuous data collection from deployed systems
- Iterative model updates

**Training:**
- Large-scale distributed training
- Online learning from production data
- A/B testing with model versions

**Goal:** Robust 24/7 performance across all scenarios

---

## **ANNOTATION STRATEGY**

### **Cost Estimation**

| Task | Time per Image | Cost per Image | Total (10K images) |
|------|----------------|----------------|-------------------|
| Bounding boxes (vehicles) | 2 min | $0.20 | $2,000 |
| Segmentation (driveway/footpath) | 5 min | $0.50 | $5,000 |
| Plate detection | 1 min | $0.10 | $1,000 |
| OCR transcription | 30 sec | $0.05 | $500 |
| Tracking IDs (video) | 10 min/video | $1.00/video | $10,000 (10K frames) |
| **TOTAL** | | | **$18,500** |

### **Annotation Tools**

- **CVAT (Computer Vision Annotation Tool):** Free, open-source
- **Labelbox:** Commercial, $500-1000/month
- **Amazon SageMaker Ground Truth:** Pay-per-annotation
- **V7 Darwin:** AI-assisted annotation (reduces time by 30%)

### **Quality Control**

$$\text{Agreement} = \frac{2 \cdot |A_1 \cap A_2|}{|A_1| + |A_2|}$$

- Dual annotation for 10% of data
- Inter-annotator agreement (IoU > 0.85 for boxes, Dice > 0.90 for masks)
- Automated consistency checks (e.g., "plate must be inside vehicle bbox")

---

## **FINAL ANSWER**

### **Can You Use Public Datasets?**

**For final deployment: NO (alone)**
**For pre-training: YES (essential)**

### **Optimal Strategy:**

$$\text{Final Model} = \text{Pre-train}_{public} \circ \text{Fine-tune}_{custom} \circ \text{Augment}_{synthetic}$$

1. **Pre-train** backbone and decoders on public datasets (COCO, Cityscapes, CCPD, etc.)
2. **Collect** 10,000-50,000 custom annotated images from your target domain
3. **Augment** with synthetic data (especially for OCR and rare scenarios)
4. **Fine-tune** end-to-end with multi-task loss on custom data
5. **Iterate** with active learning and continuous deployment feedback

### **Minimum Viable Dataset:**

- **5,000 images** (all tasks annotated) with pre-training
- **20,000 images** without pre-training (not recommended)

### **Budget Estimate:**

- Data collection + annotation: $15,000 - $50,000
- Compute (GPU training): $5,000 - $15,000
- **Total:** $20,000 - $65,000

### **Time Estimate:**

- PoC with 1K samples: 1-2 months
- Production with 10K+ samples: 6-12 months

**The custom dataset is unavoidable for your specific requirements (driveway/footpath segmentation + Indian plates + stopped vehicle alerts), but public datasets dramatically reduce the amount of custom data needed through transfer learning.**


I'll provide detailed CLRS-style algorithms for all three training stages.

---

## **STAGE 1: PRE-TRAINED BACKBONE INITIALIZATION**

### **ALGORITHM S1.1: LOAD-PRETRAINED-BACKBONE**

**Input:** Pre-trained model path $path_{pretrained}$, target architecture $\mathcal{A}$

**Output:** Initialized backbone parameters $\Theta_{backbone}$

1. **// Load pre-trained weights from ImageNet or COCO**
2. $\Theta_{pretrained} \leftarrow$ **LOAD-CHECKPOINT**$(path_{pretrained})$
3. 
4. **// Extract only backbone parameters (ResNet50)**
5. $\Theta_{backbone} \leftarrow \emptyset$
6. **for each** layer $\ell \in \{conv1, layer1, layer2, layer3, layer4\}$ **do**
7. $\quad$ **if** $\ell \in \Theta_{pretrained}$ **then**
8. $\quad\quad$ $\Theta_{backbone}[\ell] \leftarrow \Theta_{pretrained}[\ell]$
9. $\quad$ **else**
10. $\quad\quad$ **RAISE** ERROR "Missing layer in pre-trained model"
11. 
12. **// Verify dimension compatibility**
13. **for each** layer $\ell \in \Theta_{backbone}$ **do**
14. $\quad$ $expected\_shape \leftarrow \mathcal{A}.get\_shape(\ell)$
15. $\quad$ $actual\_shape \leftarrow \Theta_{backbone}[\ell].shape$
16. $\quad$ **if** $expected\_shape \neq actual\_shape$ **then**
17. $\quad\quad$ **RAISE** ERROR "Shape mismatch in layer " $\ell$
18. 
19. **// Initialize new layers (projection layers for multi-scale features)**
20. **for** $scale \in \{C_3, C_4, C_5\}$ **do**
21. $\quad$ $W_{proj}^{scale} \leftarrow$ **XAVIER-INIT**$(C_{in}^{scale}, D)$
22. $\quad$ $b_{proj}^{scale} \leftarrow$ **ZERO-INIT**$(D)$
23. $\quad$ $\Theta_{backbone}[proj\_scale] \leftarrow (W_{proj}^{scale}, b_{proj}^{scale})$
24. 
25. **return** $\Theta_{backbone}$

---

### **ALGORITHM S1.2: INITIALIZE-TRANSFORMER-ENCODER**

**Input:** Model dimension $D$, number of layers $L$, number of heads $h$

**Output:** Initialized transformer encoder parameters $\Theta_{encoder}$

1. $\Theta_{encoder} \leftarrow \emptyset$
2. $d_k \leftarrow D / h$ **// Dimension per attention head**
3. 
4. **// Initialize positional encoding (fixed, not learned)**
5. $PE \leftarrow$ **SINUSOIDAL-POSITIONAL-ENCODING**$(N_{max}, D)$
6. $\Theta_{encoder}[PE] \leftarrow PE$ **// Store but don't train**
7. 
8. **for** $\ell = 1$ **to** $L$ **do**
9. $\quad$ **// Multi-head self-attention parameters**
10. $\quad$ **for** $i = 1$ **to** $h$ **do**
11. $\quad\quad$ $W_{i,Q}^{(\ell)} \leftarrow$ **XAVIER-INIT**$(D, d_k)$
12. $\quad\quad$ $W_{i,K}^{(\ell)} \leftarrow$ **XAVIER-INIT**$(D, d_k)$
13. $\quad\quad$ $W_{i,V}^{(\ell)} \leftarrow$ **XAVIER-INIT**$(D, d_k)$
14. $\quad$
15. $\quad$ $W_O^{(\ell)} \leftarrow$ **XAVIER-INIT**$(D, D)$
16. $\quad$ $b_O^{(\ell)} \leftarrow$ **ZERO-INIT**$(D)$
17. $\quad$
18. $\quad$ **// Layer normalization parameters**
19. $\quad$ $\gamma_{LN1}^{(\ell)} \leftarrow$ **ONES-INIT**$(D)$
20. $\quad$ $\beta_{LN1}^{(\ell)} \leftarrow$ **ZERO-INIT**$(D)$
21. $\quad$ $\gamma_{LN2}^{(\ell)} \leftarrow$ **ONES-INIT**$(D)$
22. $\quad$ $\beta_{LN2}^{(\ell)} \leftarrow$ **ZERO-INIT**$(D)$
23. $\quad$
24. $\quad$ **// Feed-forward network parameters**
25. $\quad$ $W_1^{(\ell)} \leftarrow$ **XAVIER-INIT**$(D, 4D)$
26. $\quad$ $b_1^{(\ell)} \leftarrow$ **ZERO-INIT**$(4D)$
27. $\quad$ $W_2^{(\ell)} \leftarrow$ **XAVIER-INIT**$(4D, D)$
28. $\quad$ $b_2^{(\ell)} \leftarrow$ **ZERO-INIT**$(D)$
29. $\quad$
30. $\quad$ **// Store layer parameters**
31. $\quad$ $\Theta_{encoder}[layer\_\ell] \leftarrow \{W_{i,Q}^{(\ell)}, W_{i,K}^{(\ell)}, W_{i,V}^{(\ell)}, W_O^{(\ell)}, b_O^{(\ell)}, \gamma_{LN1}^{(\ell)}, \beta_{LN1}^{(\ell)}, \gamma_{LN2}^{(\ell)}, \beta_{LN2}^{(\ell)}, W_1^{(\ell)}, b_1^{(\ell)}, W_2^{(\ell)}, b_2^{(\ell)}\}$
32. 
33. **return** $\Theta_{encoder}$

---

### **ALGORITHM S1.3: INITIALIZE-ALL-DECODERS**

**Input:** Number of queries per task $\{n_{det}, n_{seg}, n_{plate}, n_{ocr}, n_{track}\}$, model dimension $D$

**Output:** Initialized decoder parameters $\{\Theta_{det}, \Theta_{seg}, \Theta_{plate}, \Theta_{ocr}, \Theta_{track}\}$

1. **// Initialize learnable query embeddings**
2. $Q_{det} \leftarrow$ **UNIFORM-INIT**$(n_{det}, D)$ range $[-\frac{1}{\sqrt{D}}, \frac{1}{\sqrt{D}}]$
3. $Q_{seg} \leftarrow$ **UNIFORM-INIT**$(n_{seg}, D)$
4. $Q_{plate} \leftarrow$ **UNIFORM-INIT**$(n_{plate}, D)$
5. $Q_{ocr} \leftarrow$ **UNIFORM-INIT**$(n_{ocr}, D)$
6. $Q_{track} \leftarrow$ **UNIFORM-INIT**$(n_{track}, D)$
7. 
8. **// Initialize each decoder independently**
9. $\Theta_{det} \leftarrow$ **INITIALIZE-SINGLE-DECODER**$(Q_{det}, D, L_{dec}=6, \text{has\_inter\_attn}=\text{False})$
10. $\Theta_{seg} \leftarrow$ **INITIALIZE-SINGLE-DECODER**$(Q_{seg}, D, L_{dec}=6, \text{has\_inter\_attn}=\text{True})$
11. $\Theta_{plate} \leftarrow$ **INITIALIZE-SINGLE-DECODER**$(Q_{plate}, D, L_{dec}=6, \text{has\_inter\_attn}=\text{True})$
12. $\Theta_{ocr} \leftarrow$ **INITIALIZE-SINGLE-DECODER**$(Q_{ocr}, D, L_{dec}=6, \text{has\_inter\_attn}=\text{True})$
13. $\Theta_{track} \leftarrow$ **INITIALIZE-SINGLE-DECODER**$(Q_{track}, D, L_{dec}=6, \text{has\_inter\_attn}=\text{True})$
14. 
15. **// Initialize task-specific output heads**
16. $\Theta_{det}[heads] \leftarrow$ **INITIALIZE-DETECTION-HEADS**$(D, n_{classes}=7, n_{colors}, n_{types})$
17. $\Theta_{seg}[heads] \leftarrow$ **INITIALIZE-SEGMENTATION-HEADS**$(D, n_{seg\_classes}=3)$
18. $\Theta_{plate}[heads] \leftarrow$ **INITIALIZE-BBOX-HEAD**$(D)$
19. $\Theta_{ocr}[heads] \leftarrow$ **INITIALIZE-OCR-HEAD**$(D, vocab\_size=|\mathcal{A}|+1)$
20. $\Theta_{track}[heads] \leftarrow$ **INITIALIZE-TRACKING-HEAD**$(D)$
21. 
22. **return** $\{\Theta_{det}, \Theta_{seg}, \Theta_{plate}, \Theta_{ocr}, \Theta_{track}\}$

---

### **ALGORITHM S1.4: INITIALIZE-SINGLE-DECODER**

**Input:** Query embeddings $Q \in \mathbb{R}^{n \times D}$, model dimension $D$, number of layers $L_{dec}$, inter-decoder attention flag $has\_inter\_attn$

**Output:** Initialized decoder parameters $\Theta_{decoder}$

1. $\Theta_{decoder} \leftarrow \emptyset$
2. $h \leftarrow 8$ **// Number of attention heads**
3. $d_k \leftarrow D / h$
4. 
5. **// Store query embeddings**
6. $\Theta_{decoder}[queries] \leftarrow Q$ **// These are learnable**
7. 
8. **for** $\ell = 1$ **to** $L_{dec}$ **do**
9. $\quad$ **// Self-attention parameters**
10. $\quad$ **for** $i = 1$ **to** $h$ **do**
11. $\quad\quad$ $W_{i,Q}^{self,(\ell)} \leftarrow$ **XAVIER-INIT**$(D, d_k)$
12. $\quad\quad$ $W_{i,K}^{self,(\ell)} \leftarrow$ **XAVIER-INIT**$(D, d_k)$
13. $\quad\quad$ $W_{i,V}^{self,(\ell)} \leftarrow$ **XAVIER-INIT**$(D, d_k)$
14. $\quad$ $W_O^{self,(\ell)} \leftarrow$ **XAVIER-INIT**$(D, D)$
15. $\quad$ $\gamma_{LN1}^{(\ell)}, \beta_{LN1}^{(\ell)} \leftarrow$ **ONES-INIT**$(D)$, **ZERO-INIT**$(D)$
16. $\quad$
17. $\quad$ **// Cross-attention with encoder memory**
18. $\quad$ **for** $i = 1$ **to** $h$ **do**
19. $\quad\quad$ $W_{i,Q}^{enc,(\ell)} \leftarrow$ **XAVIER-INIT**$(D, d_k)$
20. $\quad\quad$ $W_{i,K}^{enc,(\ell)} \leftarrow$ **XAVIER-INIT**$(D, d_k)$
21. $\quad\quad$ $W_{i,V}^{enc,(\ell)} \leftarrow$ **XAVIER-INIT**$(D, d_k)$
22. $\quad$ $W_O^{enc,(\ell)} \leftarrow$ **XAVIER-INIT**$(D, D)$
23. $\quad$ $\gamma_{LN2}^{(\ell)}, \beta_{LN2}^{(\ell)} \leftarrow$ **ONES-INIT**$(D)$, **ZERO-INIT**$(D)$
24. $\quad$
25. $\quad$ **// Inter-decoder cross-attention (if applicable)**
26. $\quad$ **if** $has\_inter\_attn = \text{True}$ **then**
27. $\quad\quad$ **for** $i = 1$ **to** $h$ **do**
28. $\quad\quad\quad$ $W_{i,Q}^{inter,(\ell)} \leftarrow$ **XAVIER-INIT**$(D, d_k)$
29. $\quad\quad\quad$ $W_{i,K}^{inter,(\ell)} \leftarrow$ **XAVIER-INIT**$(D, d_k)$
30. $\quad\quad\quad$ $W_{i,V}^{inter,(\ell)} \leftarrow$ **XAVIER-INIT**$(D, d_k)$
31. $\quad\quad$ $W_O^{inter,(\ell)} \leftarrow$ **XAVIER-INIT**$(D, D)$
32. $\quad\quad$ $\gamma_{LN3}^{(\ell)}, \beta_{LN3}^{(\ell)} \leftarrow$ **ONES-INIT**$(D)$, **ZERO-INIT**$(D)$
33. $\quad$
34. $\quad$ **// Feed-forward network**
35. $\quad$ $W_1^{(\ell)} \leftarrow$ **XAVIER-INIT**$(D, 4D)$
36. $\quad$ $b_1^{(\ell)} \leftarrow$ **ZERO-INIT**$(4D)$
37. $\quad$ $W_2^{(\ell)} \leftarrow$ **XAVIER-INIT**$(4D, D)$
38. $\quad$ $b_2^{(\ell)} \leftarrow$ **ZERO-INIT**$(D)$
39. $\quad$ $\gamma_{LN4}^{(\ell)}, \beta_{LN4}^{(\ell)} \leftarrow$ **ONES-INIT**$(D)$, **ZERO-INIT**$(D)$
40. $\quad$
41. $\quad$ **// Store all layer parameters**
42. $\quad$ $\Theta_{decoder}[layer\_\ell] \leftarrow$ **COLLECT-ALL-PARAMS**()
43. 
44. **return** $\Theta_{decoder}$

---

### **ALGORITHM S1.5: XAVIER-INIT**

**Input:** Input dimension $n_{in}$, output dimension $n_{out}$

**Output:** Initialized weight matrix $W \in \mathbb{R}^{n_{in} \times n_{out}}$

1. $limit \leftarrow \sqrt{\frac{6}{n_{in} + n_{out}}}$
2. **for** $i = 1$ **to** $n_{in}$ **do**
3. $\quad$ **for** $j = 1$ **to** $n_{out}$ **do**
4. $\quad\quad$ $W[i,j] \leftarrow$ **UNIFORM**$(-limit, limit)$
5. **return** $W$

---

## **STAGE 2: FINE-TUNE INDIVIDUAL DECODERS ON PUBLIC DATA**

### **ALGORITHM S2.1: FINE-TUNE-DETECTION-DECODER**

**Input:** Public dataset $\mathcal{D}_{COCO}$, initialized parameters $\{\Theta_{backbone}, \Theta_{encoder}, \Theta_{det}\}$, learning rate $\eta$, epochs $E$

**Output:** Fine-tuned detection decoder $\Theta_{det}^{*}$

1. **// Freeze backbone and encoder**
2. **FREEZE-PARAMETERS**$(\Theta_{backbone})$
3. **FREEZE-PARAMETERS**$(\Theta_{encoder})$
4. 
5. **// Only detection decoder and output heads are trainable**
6. $\Theta_{train} \leftarrow \Theta_{det}$
7. 
8. **// Initialize optimizer**
9. $optimizer \leftarrow$ **ADAM**$(\Theta_{train}, \eta=10^{-4}, \beta_1=0.9, \beta_2=0.999)$
10. 
11. **// Map COCO classes to target classes**
12. $class\_mapping \leftarrow$ **CREATE-CLASS-MAPPING**()
13. **// COCO: {1: person, 3: car, 6: bus, 8: truck, 4: motorcycle} → Target: {car, bus, truck, bike}**
14. 
15. **for** $epoch = 1$ **to** $E$ **do**
16. $\quad$ $\mathcal{L}_{total} \leftarrow 0$
17. $\quad$ 
18. $\quad$ **for each** batch $(I_{batch}, Y_{batch}) \in \mathcal{D}_{COCO}$ **do**
19. $\quad\quad$ **// Forward pass through frozen backbone and encoder**
20. $\quad\quad$ $C_3, C_4, C_5 \leftarrow$ **CNN-BACKBONE**$(I_{batch}; \Theta_{backbone})$ **// No gradient**
21. $\quad\quad$ $\mathcal{M} \leftarrow$ **TRANSFORMER-ENCODER**$(C_3, C_4, C_5; \Theta_{encoder})$ **// No gradient**
22. $\quad\quad$ 
23. $\quad\quad$ **// Forward pass through trainable detection decoder**
24. $\quad\quad$ $\mathcal{D}_{feat}, \mathcal{D} \leftarrow$ **DETECTION-DECODER**$(Q_{det}, \mathcal{M}, \text{null}; \Theta_{det})$
25. $\quad\quad$ 
26. $\quad\quad$ **// Remap COCO labels to target classes**
27. $\quad\quad$ $Y_{batch}^{remapped} \leftarrow$ **REMAP-LABELS**$(Y_{batch}, class\_mapping)$
28. $\quad\quad$ 
29. $\quad\quad$ **// Hungarian matching between predictions and ground truth**
30. $\quad\quad$ $matches \leftarrow$ **HUNGARIAN-MATCHING**$(\mathcal{D}, Y_{batch}^{remapped})$
31. $\quad\quad$ 
32. $\quad\quad$ **// Compute detection loss**
33. $\quad\quad$ $\mathcal{L}_{cls} \leftarrow$ **FOCAL-LOSS**$(\mathcal{D}, Y_{batch}^{remapped}, matches)$
34. $\quad\quad$ $\mathcal{L}_{bbox} \leftarrow$ **GIOU-LOSS**$(\mathcal{D}, Y_{batch}^{remapped}, matches)$
35. $\quad\quad$ $\mathcal{L}_{batch} \leftarrow \mathcal{L}_{cls} + \lambda_{bbox} \mathcal{L}_{bbox}$
36. $\quad\quad$ 
37. $\quad\quad$ **// Backward pass (only through decoder)**
38. $\quad\quad$ $\nabla_{\Theta_{det}} \mathcal{L}_{batch} \leftarrow$ **BACKPROPAGATE**$(\mathcal{L}_{batch})$
39. $\quad\quad$ 
40. $\quad\quad$ **// Update parameters**
41. $\quad\quad$ $\Theta_{det} \leftarrow$ **OPTIMIZER-STEP**$(optimizer, \Theta_{det}, \nabla_{\Theta_{det}} \mathcal{L}_{batch})$
42. $\quad\quad$ 
43. $\quad\quad$ $\mathcal{L}_{total} \leftarrow \mathcal{L}_{total} + \mathcal{L}_{batch}$
44. $\quad$
45. $\quad$ **// Log epoch statistics**
46. $\quad$ **PRINT** "Epoch", $epoch$, "Loss:", $\mathcal{L}_{total}$
47. $\quad$ 
48. $\quad$ **// Validation**
49. $\quad$ **if** $epoch \mod 5 = 0$ **then**
50. $\quad\quad$ $mAP \leftarrow$ **EVALUATE-DETECTION**$(\mathcal{D}_{val}, \Theta_{det})$
51. $\quad\quad$ **PRINT** "Validation mAP:", $mAP$
52. 
53. $\Theta_{det}^{*} \leftarrow \Theta_{det}$
54. **return** $\Theta_{det}^{*}$

---

### **ALGORITHM S2.2: HUNGARIAN-MATCHING**

**Input:** Predictions $\mathcal{D} = \{(bbox_i, class_i)\}_{i=1}^{n_{queries}}$, ground truth $Y = \{(bbox_j^{gt}, class_j^{gt})\}_{j=1}^{n_{gt}}$

**Output:** Matching indices $matches = \{(i, j)\}$ indicating query $i$ matches GT $j$

1. **// Compute cost matrix**
2. $C \leftarrow$ **ZEROS**$(n_{queries}, n_{gt})$
3. 
4. **for** $i = 1$ **to** $n_{queries}$ **do**
5. $\quad$ **for** $j = 1$ **to** $n_{gt}$ **do**
6. $\quad\quad$ **// Classification cost**
7. $\quad\quad$ $\mathcal{L}_{cls}^{ij} \leftarrow -\log(p_{class_i}[class_j^{gt}])$
8. $\quad\quad$ 
9. $\quad\quad$ **// Bounding box L1 cost**
10. $\quad\quad$ $\mathcal{L}_{L1}^{ij} \leftarrow \|bbox_i - bbox_j^{gt}\|_1$
11. $\quad\quad$ 
12. $\quad\quad$ **// GIoU cost**
13. $\quad\quad$ $\mathcal{L}_{GIoU}^{ij} \leftarrow 1 - \text{GIoU}(bbox_i, bbox_j^{gt})$
14. $\quad\quad$ 
15. $\quad\quad$ **// Total cost**
16. $\quad\quad$ $C[i,j] \leftarrow \lambda_{cls} \mathcal{L}_{cls}^{ij} + \lambda_{L1} \mathcal{L}_{L1}^{ij} + \lambda_{GIoU} \mathcal{L}_{GIoU}^{ij}$
17. 
18. **// Solve assignment problem using Hungarian algorithm**
19. $matches \leftarrow$ **SCIPY-LINEAR-SUM-ASSIGNMENT**$(C)$
20. 
21. **return** $matches$

**Note:** Hungarian algorithm finds optimal one-to-one matching in $\mathcal{O}(n^3)$ time.

---

### **ALGORITHM S2.3: FINE-TUNE-SEGMENTATION-DECODER**

**Input:** Public dataset $\mathcal{D}_{Cityscapes}$, parameters $\{\Theta_{backbone}, \Theta_{encoder}, \Theta_{seg}, \Theta_{det}\}$, learning rate $\eta$, epochs $E$

**Output:** Fine-tuned segmentation decoder $\Theta_{seg}^{*}$

1. **// Freeze backbone, encoder, and detection decoder**
2. **FREEZE-PARAMETERS**$(\Theta_{backbone})$
3. **FREEZE-PARAMETERS**$(\Theta_{encoder})$
4. **FREEZE-PARAMETERS**$(\Theta_{det})$ **// Detection already fine-tuned**
5. 
6. **// Only segmentation decoder is trainable**
7. $\Theta_{train} \leftarrow \Theta_{seg}$
8. 
9. $optimizer \leftarrow$ **ADAM**$(\Theta_{train}, \eta=10^{-4})$
10. 
11. **// Map Cityscapes classes to target classes**
12. $seg\_mapping \leftarrow$ **CREATE-SEG-MAPPING**()
13. **// Cityscapes: {road, sidewalk} → Target: {driveway, footpath}**
14. **// This is approximate; driveway ≈ road, footpath ≈ sidewalk**
15. 
16. **for** $epoch = 1$ **to** $E$ **do**
17. $\quad$ $\mathcal{L}_{total} \leftarrow 0$
18. $\quad$ 
19. $\quad$ **for each** batch $(I_{batch}, Y_{batch}^{seg}) \in \mathcal{D}_{Cityscapes}$ **do**
20. $\quad\quad$ **// Forward pass through frozen components**
21. $\quad\quad$ $C_3, C_4, C_5 \leftarrow$ **CNN-BACKBONE**$(I_{batch}; \Theta_{backbone})$
22. $\quad\quad$ $\mathcal{M} \leftarrow$ **TRANSFORMER-ENCODER**$(C_3, C_4, C_5; \Theta_{encoder})$
23. $\quad\quad$ $\mathcal{D}_{feat}, \_ \leftarrow$ **DETECTION-DECODER**$(Q_{det}, \mathcal{M}, \text{null}; \Theta_{det})$
24. $\quad\quad$ 
25. $\quad\quad$ **// Forward through trainable segmentation decoder**
26. $\quad\quad$ $\mathcal{S}_{feat}, \mathcal{S} \leftarrow$ **SEGMENTATION-DECODER**$(Q_{seg}, \mathcal{M}, \mathcal{D}_{feat}; \Theta_{seg})$
27. $\quad\quad$ 
28. $\quad\quad$ **// Remap segmentation labels**
29. $\quad\quad$ $Y_{batch}^{remapped} \leftarrow$ **REMAP-SEG-LABELS**$(Y_{batch}^{seg}, seg\_mapping)$
30. $\quad\quad$ 
31. $\quad\quad$ **// Compute segmentation loss**
32. $\quad\quad$ $\mathcal{L}_{CE} \leftarrow$ **CROSS-ENTROPY-LOSS**$(\mathcal{S}, Y_{batch}^{remapped})$
33. $\quad\quad$ $\mathcal{L}_{Dice} \leftarrow$ **DICE-LOSS**$(\mathcal{S}, Y_{batch}^{remapped})$
34. $\quad\quad$ $\mathcal{L}_{batch} \leftarrow \mathcal{L}_{CE} + \mathcal{L}_{Dice}$
35. $\quad\quad$ 
36. $\quad\quad$ **// Backward and update**
37. $\quad\quad$ $\nabla_{\Theta_{seg}} \mathcal{L}_{batch} \leftarrow$ **BACKPROPAGATE**$(\mathcal{L}_{batch})$
38. $\quad\quad$ $\Theta_{seg} \leftarrow$ **OPTIMIZER-STEP**$(optimizer, \Theta_{seg}, \nabla_{\Theta_{seg}} \mathcal{L}_{batch})$
39. $\quad\quad$ 
40. $\quad\quad$ $\mathcal{L}_{total} \leftarrow \mathcal{L}_{total} + \mathcal{L}_{batch}$
41. $\quad$
42. $\quad$ **PRINT** "Epoch", $epoch$, "Seg Loss:", $\mathcal{L}_{total}$
43. $\quad$
44. $\quad$ **if** $epoch \mod 5 = 0$ **then**
45. $\quad\quad$ $mIoU \leftarrow$ **EVALUATE-SEGMENTATION**$(\mathcal{D}_{val}, \Theta_{seg})$
46. $\quad\quad$ **PRINT** "Validation mIoU:", $mIoU$
47. 
48. $\Theta_{seg}^{*} \leftarrow \Theta_{seg}$
49. **return** $\Theta_{seg}^{*}$

---

### **ALGORITHM S2.4: FINE-TUNE-PLATE-DECODER**

**Input:** Public dataset $\mathcal{D}_{CCPD}$ (Chinese City Parking Dataset), parameters $\{\Theta_{backbone}, \Theta_{encoder}, \Theta_{det}, \Theta_{plate}\}$, learning rate $\eta$, epochs $E$

**Output:** Fine-tuned plate decoder $\Theta_{plate}^{*}$

1. **// Freeze all except plate decoder**
2. **FREEZE-PARAMETERS**$(\Theta_{backbone})$
3. **FREEZE-PARAMETERS**$(\Theta_{encoder})$
4. **FREEZE-PARAMETERS**$(\Theta_{det})$
5. 
6. $\Theta_{train} \leftarrow \Theta_{plate}$
7. $optimizer \leftarrow$ **ADAM**$(\Theta_{train}, \eta=10^{-4})$
8. 
9. **for** $epoch = 1$ **to** $E$ **do**
10. $\quad$ $\mathcal{L}_{total} \leftarrow 0$
11. $\quad$ 
12. $\quad$ **for each** batch $(I_{batch}, Y_{batch}^{plate}) \in \mathcal{D}_{CCPD}$ **do**
13. $\quad\quad$ **// Forward through frozen components**
14. $\quad\quad$ $C_3, C_4, C_5 \leftarrow$ **CNN-BACKBONE**$(I_{batch}; \Theta_{backbone})$
15. $\quad\quad$ $\mathcal{M} \leftarrow$ **TRANSFORMER-ENCODER**$(C_3, C_4, C_5; \Theta_{encoder})$
16. $\quad\quad$ $\mathcal{D}_{feat}, \_ \leftarrow$ **DETECTION-DECODER**$(Q_{det}, \mathcal{M}, \text{null}; \Theta_{det})$
17. $\quad\quad$ 
18. $\quad\quad$ **// Forward through trainable plate decoder**
19. $\quad\quad$ $\mathcal{P}_{feat}, \mathcal{P} \leftarrow$ **PLATE-DECODER**$(Q_{plate}, \mathcal{M}, \mathcal{D}_{feat}; \Theta_{plate})$
20. $\quad\quad$ 
21. $\quad\quad$ **// Hungarian matching for plate detection**
22. $\quad\quad$ $matches \leftarrow$ **HUNGARIAN-MATCHING**$(\mathcal{P}, Y_{batch}^{plate})$
23. $\quad\quad$ 
24. $\quad\quad$ **// Compute plate detection loss**
25. $\quad\quad$ $\mathcal{L}_{cls}^{plate} \leftarrow$ **FOCAL-LOSS**$(\mathcal{P}, Y_{batch}^{plate}, matches)$
26. $\quad\quad$ $\mathcal{L}_{bbox}^{plate} \leftarrow$ **GIOU-LOSS**$(\mathcal{P}, Y_{batch}^{plate}, matches)$
27. $\quad\quad$ $\mathcal{L}_{batch} \leftarrow \mathcal{L}_{cls}^{plate} + \lambda_{bbox} \mathcal{L}_{bbox}^{plate}$
28. $\quad\quad$ 
29. $\quad\quad$ **// Backward and update**
30. $\quad\quad$ $\nabla_{\Theta_{plate}} \mathcal{L}_{batch} \leftarrow$ **BACKPROPAGATE**$(\mathcal{L}_{batch})$
31. $\quad\quad$ $\Theta_{plate} \leftarrow$ **OPTIMIZER-STEP**$(optimizer, \Theta_{plate}, \nabla_{\Theta_{plate}} \mathcal{L}_{batch})$
32. $\quad\quad$ 
33. $\quad\quad$ $\mathcal{L}_{total} \leftarrow \mathcal{L}_{total} + \mathcal{L}_{batch}$
34. $\quad$
35. $\quad$ **PRINT** "Epoch", $epoch$, "Plate Loss:", $\mathcal{L}_{total}$
36. $\quad$
37. $\quad$ **if** $epoch \mod 5 = 0$ **then**
38. $\quad\quad$ $mAP_{plate} \leftarrow$ **EVALUATE-PLATE-DETECTION**$(\mathcal{D}_{val}, \Theta_{plate})$
39. $\quad\quad$ **PRINT** "Validation Plate mAP:", $mAP_{plate}$
40. 
41. $\Theta_{plate}^{*} \leftarrow \Theta_{plate}$
42. **return** $\Theta_{plate}^{*}$

---

### **ALGORITHM S2.5: FINE-TUNE-OCR-DECODER**

**Input:** Mixed dataset $\mathcal{D}_{OCR} = \mathcal{D}_{CCPD} \cup \mathcal{D}_{synthetic}$, parameters $\{\Theta_{backbone}, \Theta_{encoder}, \Theta_{plate}, \Theta_{ocr}\}$, learning rate $\eta$, epochs $E$

**Output:** Fine-tuned OCR decoder $\Theta_{ocr}^{*}$

1. **// Freeze all except OCR decoder**
2. **FREEZE-PARAMETERS**$(\Theta_{backbone}, \Theta_{encoder}, \Theta_{plate})$
3. 
4. $\Theta_{train} \leftarrow \Theta_{ocr}$
5. $optimizer \leftarrow$ **ADAM**$(\Theta_{train}, \eta=10^{-4})$
6. 
7. **// Generate synthetic Indian plates**
8. $\mathcal{D}_{synthetic} \leftarrow$ **GENERATE-SYNTHETIC-PLATES**$(n_{synthetic}=50000)$
9. $\mathcal{D}_{OCR} \leftarrow \mathcal{D}_{CCPD} \cup \mathcal{D}_{synthetic}$
10. 
11. **for** $epoch = 1$ **to** $E$ **do**
12. $\quad$ $\mathcal{L}_{total} \leftarrow 0$
13. $\quad$ 
14. $\quad$ **for each** batch $(I_{batch}, Y_{batch}^{text}) \in \mathcal{D}_{OCR}$ **do**
15. $\quad\quad$ **// Forward through frozen components**
16. $\quad\quad$ $C_3, C_4, C_5 \leftarrow$ **CNN-BACKBONE**$(I_{batch}; \Theta_{backbone})$
17. $\quad\quad$ $\mathcal{M} \leftarrow$ **TRANSFORMER-ENCODER**$(C_3, C_4, C_5; \Theta_{encoder})$
18. $\quad\quad$ $\mathcal{P}_{feat}, \_ \leftarrow$ **PLATE-DECODER**$(Q_{plate}, \mathcal{M}, \mathcal{D}_{feat}; \Theta_{plate})$
19. $\quad\quad$ 
20. $\quad\quad$ **// Forward through trainable OCR decoder**
21. $\quad\quad$ $\mathcal{O} \leftarrow$ **OCR-DECODER**$(Q_{ocr}, \mathcal{M}, \mathcal{P}_{feat}; \Theta_{ocr})$
22. $\quad\quad$ 
23. $\quad\quad$ **// Compute CTC loss**
24. $\quad\quad$ $\mathcal{L}_{CTC} \leftarrow$ **CTC-LOSS**$(\mathcal{O}, Y_{batch}^{text})$
25. $\quad\quad$ $\mathcal{L}_{batch} \leftarrow \mathcal{L}_{CTC}$
26. $\quad\quad$ 
27. $\quad\quad$ **// Backward and update**
28. $\quad\quad$ $\nabla_{\Theta_{ocr}} \mathcal{L}_{batch} \leftarrow$ **BACKPROPAGATE**$(\mathcal{L}_{batch})$
29. $\quad\quad$ $\Theta_{ocr} \leftarrow$ **OPTIMIZER-STEP**$(optimizer, \Theta_{ocr}, \nabla_{\Theta_{ocr}} \mathcal{L}_{batch})$
30. $\quad\quad$ 
31. $\quad\quad$ $\mathcal{L}_{total} \leftarrow \mathcal{L}_{total} + \mathcal{L}_{batch}$
32. $\quad$
33. $\quad$ **PRINT** "Epoch", $epoch$, "OCR Loss:", $\mathcal{L}_{total}$
34. $\quad$
35. $\quad$ **if** $epoch \mod 5 = 0$ **then**
36. $\quad\quad$ $accuracy \leftarrow$ **EVALUATE-OCR**$(\mathcal{D}_{val}, \Theta_{ocr})$
37. $\quad\quad$ **PRINT** "Validation OCR Accuracy:", $accuracy$
38. 
39. $\Theta_{ocr}^{*} \leftarrow \Theta_{ocr}$
40. **return** $\Theta_{ocr}^{*}$

---

### **ALGORITHM S2.6: GENERATE-SYNTHETIC-PLATES**

**Input:** Number of samples $n_{synthetic}$, Indian plate format regex

**Output:** Synthetic dataset $\mathcal{D}_{synthetic} = \{(I_i^{plate}, text_i)\}_{i=1}^{n_{synthetic}}$

1. $\mathcal{D}_{synthetic} \leftarrow \emptyset$
2. 
3. **// Indian plate format: [A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}**
4. $alphabet \leftarrow$ ["A", "B", ..., "Z"]
5. $digits \leftarrow$ ["0", "1", ..., "9"]
6. 
7. **// Load plate fonts and backgrounds**
8. $fonts \leftarrow$ **LOAD-PLATE-FONTS**() **// 10-15 realistic fonts**
9. $backgrounds \leftarrow$ **LOAD-BACKGROUNDS**() **// White, yellow**
10. 
11. **for** $i = 1$ **to** $n_{synthetic}$ **do**
12. $\quad$ **// Generate random plate text**
13. $\quad$ $state\_code \leftarrow$ **RANDOM-CHOICE**$(alphabet, 2)$ **// e.g., "DL"**
14. $\quad$ $district\_code \leftarrow$ **RANDOM-CHOICE**$(digits, 2)$ **// e.g., "01"**
15. $\quad$ $series \leftarrow$ **RANDOM-CHOICE**$(alphabet, 2)$ **// e.g., "AB"**
16. $\quad$ $number \leftarrow$ **RANDOM-CHOICE**$(digits, 4)$ **// e.g., "1234"**
17. $\quad$ $text_i \leftarrow state\_code + district\_code + series + number$ **// "DL01AB1234"**
18. $\quad$
19. $\quad$ **// Render plate image**
20. $\quad$ $font \leftarrow$ **RANDOM-CHOICE**$(fonts)$
21. $\quad$ $background \leftarrow$ **RANDOM-CHOICE**$(backgrounds)$
22. $\quad$ $I_i^{clean} \leftarrow$ **RENDER-TEXT**$(text_i, font, background)$
23. $\quad$
24. $\quad$ **// Apply realistic distortions**
25. $\quad$ $\theta_{rot} \leftarrow$ **UNIFORM**$(-15°, 15°)$ **// Rotation**
26. $\quad$ $I_i^{rot} \leftarrow$ **ROTATE**$(I_i^{clean}, \theta_{rot})$
27. $\quad$
28. $\quad$ $perspective \leftarrow$ **RANDOM-PERSPECTIVE-TRANSFORM**()
29. $\quad$ $I_i^{persp} \leftarrow$ **APPLY-PERSPECTIVE**$(I_i^{rot}, perspective)$
30. $\quad$
31. $\quad$ **// Add noise and blur**
32. $\quad$ $\sigma_{blur} \leftarrow$ **UNIFORM**$(0, 2)$
33. $\quad$ $I_i^{blur} \leftarrow$ **GAUSSIAN-BLUR**$(I_i^{persp}, \sigma_{blur})$
34. $\quad$
35. $\quad$ $noise \leftarrow$ **GAUSSIAN-NOISE**$(0, \sigma_{noise}=0.01)$
36. $\quad$ $I_i^{plate} \leftarrow$ **CLIP**$(I_i^{blur} + noise, 0, 1)$
37. $\quad$
38. $\quad$ **// Add to dataset**
39. $\quad$ $\mathcal{D}_{synthetic} \leftarrow \mathcal{D}_{synthetic} \cup \{(I_i^{plate}, text_i)\}$
40. 
41. **return** $\mathcal{D}_{synthetic}$

---

### **ALGORITHM S2.7: FINE-TUNE-TRACKING-DECODER**

**Input:** Public dataset $\mathcal{D}_{MOT} \cup \mathcal{D}_{BDD100K}$, parameters $\{\Theta_{backbone}, \Theta_{encoder}, \Theta_{det}, \Theta_{seg}, \Theta_{track}\}$, learning rate $\eta$, epochs $E$

**Output:** Fine-tuned tracking decoder $\Theta_{track}^{*}$

1. **// Freeze all except tracking decoder**
2. **FREEZE-PARAMETERS**$(\Theta_{backbone}, \Theta_{encoder}, \Theta_{det}, \Theta_{seg})$
3. 
4. $\Theta_{train} \leftarrow \Theta_{track}$
5. $optimizer \leftarrow$ **ADAM**$(\Theta_{train}, \eta=10^{-4})$
6. 
7. **for** $epoch = 1$ **to** $E$ **do**
8. $\quad$ $\mathcal{L}_{total} \leftarrow 0$
9. $\quad$ 
10. $\quad$ **for each** video sequence $(V, Y^{track}) \in \mathcal{D}_{MOT}$ **do**
11. $\quad\quad$ $T_{prev} \leftarrow$ **INIT-EMPTY-TRACKS**()
12. $\quad\quad$
13. $\quad\quad$ **for** $t = 1$ **to** $|V|$ **do** **// Process frame by frame**
14. $\quad\quad\quad$ $I_t \leftarrow V[t]$ **// Frame at time t**
15. $\quad\quad\quad$ $Y_t^{track} \leftarrow Y^{track}[t]$ **// Ground truth tracks**
16. $\quad\quad\quad$
17. $\quad\quad\quad$ **// Forward through frozen components**
18. $\quad\quad\quad$ $C_3, C_4, C_5 \leftarrow$ **CNN-BACKBONE**$(I_t; \Theta_{backbone})$
19. $\quad\quad\quad$ $\mathcal{M} \leftarrow$ **TRANSFORMER-ENCODER**$(C_3, C_4, C_5; \Theta_{encoder})$
20. $\quad\quad\quad$ $\mathcal{D}_{feat}, \mathcal{D} \leftarrow$ **DETECTION-DECODER**$(Q_{det}, \mathcal{M}, \text{null}; \Theta_{det})$
21. $\quad\quad\quad$ $\mathcal{S}_{feat}, \_ \leftarrow$ **SEGMENTATION-DECODER**$(Q_{seg}, \mathcal{M}, \mathcal{D}_{feat}; \Theta_{seg})$
22. $\quad\quad\quad$
23. $\quad\quad\quad$ **// Forward through trainable tracking decoder**
24. $\quad\quad\quad$ $\mathcal{T}_t \leftarrow$ **TRACKING-DECODER**$(Q_{track}, \mathcal{M}, \mathcal{D}_{feat}, \mathcal{S}_{feat}, T_{prev}; \Theta_{track})$
25. $\quad\quad\quad$
26. $\quad\quad\quad$ **// Compute tracking loss**
27. $\quad\quad\quad$ $\mathcal{L}_{assoc} \leftarrow$ **ASSOCIATION-LOSS**$(\mathcal{T}_t, Y_t^{track})$
28. $\quad\quad\quad$ $\mathcal{L}_{traj} \leftarrow$ **TRAJECTORY-LOSS**$(\mathcal{T}_t, Y_t^{track})$
29. $\quad\quad\quad$ $\mathcal{L}_t \leftarrow \mathcal{L}_{assoc} + \lambda_{traj} \mathcal{L}_{traj}$
30. $\quad\quad\quad$
31. $\quad\quad\quad$ **// Backward and update**
32. $\quad\quad\quad$ $\nabla_{\Theta_{track}} \mathcal{L}_t \leftarrow$ **BACKPROPAGATE**$(\mathcal{L}_t)$
33. $\quad\quad\quad$ $\Theta_{track} \leftarrow$ **OPTIMIZER-STEP**$(optimizer, \Theta_{track}, \nabla_{\Theta_{track}} \mathcal{L}_t)$
34. $\quad\quad\quad$
35. $\quad\quad\quad$ **// Update previous tracks for next frame**
36. $\quad\quad\quad$ $T_{prev} \leftarrow \mathcal{T}_t$
37. $\quad\quad\quad$
38. $\quad\quad\quad$ $\mathcal{L}_{total} \leftarrow \mathcal{L}_{total} + \mathcal{L}_t$
39. $\quad$
40. $\quad$ **PRINT** "Epoch", $epoch$, "Tracking Loss:", $\mathcal{L}_{total}$
41. $\quad$
42. $\quad$ **if** $epoch \mod 5 = 0$ **then**
43. $\quad\quad$ $MOTA \leftarrow$ **EVALUATE-TRACKING**$(\mathcal{D}_{val}, \Theta_{track})$
44. $\quad\quad$ **PRINT** "Validation MOTA:", $MOTA$
45. 
46. $\Theta_{track}^{*} \leftarrow \Theta_{track}$
47. **return** $\Theta_{track}^{*}$

---

## **STAGE 3: JOINT TRAINING ON CUSTOM DATASET**

### **ALGORITHM S3.1: JOINT-MULTI-TASK-TRAINING**

**Input:** Custom dataset $\mathcal{D}_{custom} = \{(I_i, Y_i^{det}, Y_i^{seg}, Y_i^{plate}, Y_i^{ocr}, Y_i^{track})\}_{i=1}^{N}$, pre-trained parameters from Stage 2, learning rate $\eta$, epochs $E$, loss weights $\{\lambda_1, \lambda_2, \lambda_3, \lambda_4, \lambda_5\}$

**Output:** Jointly trained model $\Theta^{*} = \{\Theta_{backbone}^{*}, \Theta_{encoder}^{*}, \Theta_{det}^{*}, \Theta_{seg}^{*}, \Theta_{plate}^{*}, \Theta_{ocr}^{*}, \Theta_{track}^{*}\}$

1. **// Load all pre-trained parameters from Stage 2**
2. $\Theta \leftarrow$ **LOAD-STAGE2-CHECKPOINT**()
3. 
4. **// Unfreeze specific components based on strategy**
5. **// Strategy A: Unfreeze all (full fine-tuning)**
6. **// Strategy B: Keep backbone frozen (faster, less overfitting)**
7. **// Strategy C: Gradually unfreeze (conservative)**
8. 
9. $strategy \leftarrow$ "STRATEGY-B" **// Keep backbone frozen**
10. 
11. **if** $strategy = $ "STRATEGY-A" **then**
12. $\quad$ $\Theta_{train} \leftarrow \Theta$ **// All parameters trainable**
13. **else if** $strategy = $ "STRATEGY-B" **then**
14. $\quad$ **FREEZE-PARAMETERS**$(\Theta_{backbone})$
15. $\quad$ $\Theta_{train} \leftarrow \{\Theta_{encoder}, \Theta_{det}, \Theta_{seg}, \Theta_{plate}, \Theta_{ocr}, \Theta_{track}\}$
16. **else if** $strategy = $ "STRATEGY-C" **then**
17. $\quad$ **FREEZE-PARAMETERS**$(\Theta_{backbone}, \Theta_{encoder})$
18. $\quad$ $\Theta_{train} \leftarrow \{\Theta_{det}, \Theta_{seg}, \Theta_{plate}, \Theta_{ocr}, \Theta_{track}\}$
19. 
20. **// Initialize optimizer with different learning rates for different components**
21. $optimizer \leftarrow$ **ADAM-WITH-LAYER-LR**$(\Theta_{train})$
22. **// Backbone (if unfrozen): 1e-5, Encoder: 1e-4, Decoders: 1e-4**
23. 
24. **// Learning rate scheduler**
25. $scheduler \leftarrow$ **COSINE-ANNEALING**$(optimizer, T_{max}=E)$
26. 
27. **// Split custom dataset**
28. $\mathcal{D}_{train}, \mathcal{D}_{val} \leftarrow$ **TRAIN-VAL-SPLIT**$(\mathcal{D}_{custom}, ratio=0.9)$
29. 
30. $best\_loss \leftarrow \infty$
31. $patience \leftarrow 10$ **// Early stopping patience**
32. $epochs\_without\_improvement \leftarrow 0$
33. 
34. **for** $epoch = 1$ **to** $E$ **do**
35. $\quad$ $\mathcal{L}_{total}^{epoch} \leftarrow 0$
36. $\quad$ $\mathcal{L}_{det}^{epoch}, \mathcal{L}_{seg}^{epoch}, \mathcal{L}_{plate}^{epoch}, \mathcal{L}_{ocr}^{epoch}, \mathcal{L}_{track}^{epoch} \leftarrow 0, 0, 0, 0, 0$
37. $\quad$
38. $\quad$ **// Training loop**
39. $\quad$ **for each** batch $(I_{batch}, Y_{batch}) \in \mathcal{D}_{train}$ **do**
40. $\quad\quad$ **// Extract all ground truth annotations**
41. $\quad\quad$ $Y_{batch}^{det}, Y_{batch}^{seg}, Y_{batch}^{plate}, Y_{batch}^{ocr}, Y_{batch}^{track} \leftarrow Y_{batch}$
42. $\quad\quad$
43. $\quad\quad$ **// Full forward pass through entire architecture**
44. $\quad\quad$ $\mathcal{D}, \mathcal{S}, \mathcal{P}, \mathcal{O}, \mathcal{T}, \mathcal{A} \leftarrow$ **UNIFIED-MULTI-TASK-TRANSFORMER**$(I_{batch}, T_{prev}; \Theta)$
45. $\quad\quad$
46. $\quad\quad$ **// Compute individual task losses**
47. $\quad\quad$ $\mathcal{L}_{det}^{batch} \leftarrow$ **DETECTION-LOSS**$(\mathcal{D}, Y_{batch}^{det})$
48. $\quad\quad$ $\mathcal{L}_{seg}^{batch} \leftarrow$ **SEGMENTATION-LOSS**$(\mathcal{S}, Y_{batch}^{seg})$
49. $\quad\quad$ $\mathcal{L}_{plate}^{batch} \leftarrow$ **PLATE-LOSS**$(\mathcal{P}, Y_{batch}^{plate})$
50. $\quad\quad$ $\mathcal{L}_{ocr}^{batch} \leftarrow$ **CTC-LOSS**$(\mathcal{O}, Y_{batch}^{ocr})$
51. $\quad\quad$ $\mathcal{L}_{track}^{batch} \leftarrow$ **TRACKING-LOSS**$(\mathcal{T}, Y_{batch}^{track})$
52. $\quad\quad$
53. $\quad\quad$ **// Compute weighted multi-task loss**
54. $\quad\quad$ $\mathcal{L}_{batch} \leftarrow \lambda_1 \mathcal{L}_{det}^{batch} + \lambda_2 \mathcal{L}_{seg}^{batch} + \lambda_3 \mathcal{L}_{plate}^{batch} + \lambda_4 \mathcal{L}_{ocr}^{batch} + \lambda_5 \mathcal{L}_{track}^{batch}$
55. $\quad\quad$
56. $\quad\quad$ **// Backward pass through entire network**
57. $\quad\quad$ $\nabla_{\Theta_{train}} \mathcal{L}_{batch} \leftarrow$ **BACKPROPAGATE**$(\mathcal{L}_{batch})$
58. $\quad\quad$
59. $\quad\quad$ **// Gradient clipping to prevent explosion**
60. $\quad\quad$ $\nabla_{\Theta_{train}} \mathcal{L}_{batch} \leftarrow$ **CLIP-GRAD-NORM**$(\nabla_{\Theta_{train}} \mathcal{L}_{batch}, max\_norm=1.0)$
61. $\quad\quad$
62. $\quad\quad$ **// Update parameters**
63. $\quad\quad$ $\Theta_{train} \leftarrow$ **OPTIMIZER-STEP**$(optimizer, \Theta_{train}, \nabla_{\Theta_{train}} \mathcal{L}_{batch})$
64. $\quad\quad$
65. $\quad\quad$ **// Accumulate losses**
66. $\quad\quad$ $\mathcal{L}_{total}^{epoch} \leftarrow \mathcal{L}_{total}^{epoch} + \mathcal{L}_{batch}$
67. $\quad\quad$ $\mathcal{L}_{det}^{epoch} \leftarrow \mathcal{L}_{det}^{epoch} + \mathcal{L}_{det}^{batch}$
68. $\quad\quad$ $\mathcal{L}_{seg}^{epoch} \leftarrow \mathcal{L}_{seg}^{epoch} + \mathcal{L}_{seg}^{batch}$
69. $\quad\quad$ $\mathcal{L}_{plate}^{epoch} \leftarrow \mathcal{L}_{plate}^{epoch} + \mathcal{L}_{plate}^{batch}$
70. $\quad\quad$ $\mathcal{L}_{ocr}^{epoch} \leftarrow \mathcal{L}_{ocr}^{epoch} + \mathcal{L}_{ocr}^{batch}$
71. $\quad\quad$ $\mathcal{L}_{track}^{epoch} \leftarrow \mathcal{L}_{track}^{epoch} + \mathcal{L}_{track}^{batch}$
72. $\quad$
73. $\quad$ **// Update learning rate**
74. $\quad$ **SCHEDULER-STEP**$(scheduler)$
75. $\quad$
76. $\quad$ **// Log epoch statistics**
77. $\quad$ **PRINT** "Epoch", $epoch$, "Total Loss:", $\mathcal{L}_{total}^{epoch}$
78. $\quad$ **PRINT** "  Detection:", $\mathcal{L}_{det}^{epoch}$, "Segmentation:", $\mathcal{L}_{seg}^{epoch}$
79. $\quad$ **PRINT** "  Plate:", $\mathcal{L}_{plate}^{epoch}$, "OCR:", $\mathcal{L}_{ocr}^{epoch}$, "Tracking:", $\mathcal{L}_{track}^{epoch}$
80. $\quad$
81. $\quad$ **// Validation every epoch**
82. $\quad$ $metrics \leftarrow$ **VALIDATE-ALL-TASKS**$(\mathcal{D}_{val}, \Theta)$
83. $\quad$ **PRINT** "Validation Metrics:", $metrics$
84. $\quad$
85. $\quad$ **// Early stopping and checkpointing**
86. $\quad$ **if** $\mathcal{L}_{total}^{epoch} < best\_loss$ **then**
87. $\quad\quad$ $best\_loss \leftarrow \mathcal{L}_{total}^{epoch}$
88. $\quad\quad$ **SAVE-CHECKPOINT**$(\Theta, path=$ "best\_model.pth")
89. $\quad\quad$ $epochs\_without\_improvement \leftarrow 0$
90. $\quad$ **else**
91. $\quad\quad$ $epochs\_without\_improvement \leftarrow epochs\_without\_improvement + 1$
92. $\quad\quad$ **if** $epochs\_without\_improvement \geq patience$ **then**
93. $\quad\quad\quad$ **PRINT** "Early stopping triggered"
94. $\quad\quad\quad$ **break**
95. 
96. **// Load best model**
97. $\Theta^{*} \leftarrow$ **LOAD-CHECKPOINT**("best\_model.pth")
98. 
99. **return** $\Theta^{*}$

---

### **ALGORITHM S3.2: VALIDATE-ALL-TASKS**

**Input:** Validation dataset $\mathcal{D}_{val}$, model parameters $\Theta$

**Output:** Metrics dictionary $metrics = \{mAP_{det}, mIoU_{seg}, mAP_{plate}, accuracy_{ocr}, MOTA_{track}\}$

1. **// Set model to evaluation mode (disable dropout, etc.)**
2. **SET-EVAL-MODE**$(\Theta)$
3. 
4. **// Initialize metric accumulators**
5. $predictions_{det}, ground\_truth_{det} \leftarrow [], []$
6. $predictions_{seg}, ground\_truth_{seg} \leftarrow [], []$
7. $predictions_{plate}, ground\_truth_{plate} \leftarrow [], []$
8. $predictions_{ocr}, ground\_truth_{ocr} \leftarrow [], []$
9. $predictions_{track}, ground\_truth_{track} \leftarrow [], []$
10. 
11. **// Inference on validation set**
12. **for each** $(I, Y) \in \mathcal{D}_{val}$ **do**
13. $\quad$ **// Forward pass (no gradient computation)**
14. $\quad$ **with** **NO-GRAD**() **do**
15. $\quad\quad$ $\mathcal{D}, \mathcal{S}, \mathcal{P}, \mathcal{O}, \mathcal{T}, \mathcal{A} \leftarrow$ **UNIFIED-MULTI-TASK-TRANSFORMER**$(I, T_{prev}; \Theta)$
16. $\quad$
17. $\quad$ **// Collect predictions and ground truth**
18. $\quad$ $predictions_{det}$.append$(\mathcal{D})$
19. $\quad$ $ground\_truth_{det}$.append$(Y^{det})$
20. $\quad$ $predictions_{seg}$.append$(\mathcal{S})$
21. $\quad$ $ground\_truth_{seg}$.append$(Y^{seg})$
22. $\quad$ $predictions_{plate}$.append$(\mathcal{P})$
23. $\quad$ $ground\_truth_{plate}$.append$(Y^{plate})$
24. $\quad$ $predictions_{ocr}$.append$(\mathcal{O})$
25. $\quad$ $ground\_truth_{ocr}$.append$(Y^{ocr})$
26. $\quad$ $predictions_{track}$.append$(\mathcal{T})$
27. $\quad$ $ground\_truth_{track}$.append$(Y^{track})$
28. 
29. **// Compute metrics for each task**
30. $mAP_{det} \leftarrow$ **COMPUTE-MAP**$(predictions_{det}, ground\_truth_{det}, IoU\_threshold=0.5)$
31. $mIoU_{seg} \leftarrow$ **COMPUTE-MIOU**$(predictions_{seg}, ground\_truth_{seg})$
32. $mAP_{plate} \leftarrow$ **COMPUTE-MAP**$(predictions_{plate}, ground\_truth_{plate}, IoU\_threshold=0.5)$
33. $accuracy_{ocr} \leftarrow$ **COMPUTE-OCR-ACCURACY**$(predictions_{ocr}, ground\_truth_{ocr})$
34. $MOTA_{track} \leftarrow$ **COMPUTE-MOTA**$(predictions_{track}, ground\_truth_{track})$
35. 
36. **// Aggregate into metrics dictionary**
37. $metrics \leftarrow \{$
38. $\quad$ "detection\_mAP": $mAP_{det}$,
39. $\quad$ "segmentation\_mIoU": $mIoU_{seg}$,
40. $\quad$ "plate\_mAP": $mAP_{plate}$,
41. $\quad$ "ocr\_accuracy": $accuracy_{ocr}$,
42. $\quad$ "tracking\_MOTA": $MOTA_{track}$
43. $\}$
44. 
45. **// Set model back to training mode**
46. **SET-TRAIN-MODE**$(\Theta)$
47. 
48. **return** $metrics$

---

### **ALGORITHM S3.3: ADAPTIVE-LOSS-WEIGHTING** (Optional Enhancement)

**Input:** Task losses $\{\mathcal{L}_1, \mathcal{L}_2, \mathcal{L}_3, \mathcal{L}_4, \mathcal{L}_5\}$, learnable uncertainty parameters $\{\sigma_1, \sigma_2, \sigma_3, \sigma_4, \sigma_5\}$

**Output:** Weighted total loss $\mathcal{L}_{total}$

1. **// Uncertainty-based weighting (Kendall et al., 2018)**
2. **// Each task has a learnable uncertainty parameter**
3. 
4. $\mathcal{L}_{total} \leftarrow 0$
5. 
6. **for** $i = 1$ **to** 5 **do**
7. $\quad$ **// Weight inversely proportional to squared uncertainty**
8. $\quad$ $w_i \leftarrow \frac{1}{2\sigma_i^2}$
9. $\quad$
10. $\quad$ **// Add weighted loss**
11. $\quad$ $\mathcal{L}_{total} \leftarrow \mathcal{L}_{total} + w_i \mathcal{L}_i$
12. $\quad$
13. $\quad$ **// Add regularization term (log uncertainty)**
14. $\quad$ $\mathcal{L}_{total} \leftarrow \mathcal{L}_{total} + \log(\sigma_i)$
15. 
16. **// Mathematical form:**
17. **// $\mathcal{L}_{total} = \sum_{i=1}^5 \left[\frac{1}{2\sigma_i^2} \mathcal{L}_i + \log(\sigma_i)\right]$**
18. 
19. **return** $\mathcal{L}_{total}$

**Note:** $\sigma_i$ parameters are learned during training, automatically balancing tasks based on their inherent uncertainty.

---

### **ALGORITHM S3.4: GRADUAL-UNFREEZING** (Strategy C Detail)

**Input:** Pre-trained model $\Theta$, custom dataset $\mathcal{D}_{custom}$, total epochs $E$

**Output:** Fine-tuned model $\Theta^{*}$

1. **// Phase 1: Train only decoder output heads (Epochs 1-10)**
2. **FREEZE-PARAMETERS**$(\Theta_{backbone}, \Theta_{encoder}, \Theta_{decoder\_layers})$
3. $\Theta_{train} \leftarrow \{\Theta_{output\_heads}\}$ **// Only final prediction layers**
4. 
5. **for** $epoch = 1$ **to** $\lfloor 0.1E \rfloor$ **do**
6. $\quad$ **TRAIN-ONE-EPOCH**$(\mathcal{D}_{custom}, \Theta, \Theta_{train}, \eta=10^{-3})$
7. 
8. **// Phase 2: Unfreeze decoder layers (Epochs 11-50)**
9. **UNFREEZE-PARAMETERS**$(\Theta_{decoder\_layers})$
10. $\Theta_{train} \leftarrow \{\Theta_{decoder\_layers}, \Theta_{output\_heads}\}$
11. 
12. **for** $epoch = \lfloor 0.1E \rfloor + 1$ **to** $\lfloor 0.5E \rfloor$ **do**
13. $\quad$ **TRAIN-ONE-EPOCH**$(\mathcal{D}_{custom}, \Theta, \Theta_{train}, \eta=10^{-4})$
14. 
15. **// Phase 3: Unfreeze encoder (Epochs 51-80)**
16. **UNFREEZE-PARAMETERS**$(\Theta_{encoder})$
17. $\Theta_{train} \leftarrow \{\Theta_{encoder}, \Theta_{decoder\_layers}, \Theta_{output\_heads}\}$
18. 
19. **for** $epoch = \lfloor 0.5E \rfloor + 1$ **to** $\lfloor 0.8E \rfloor$ **do**
20. $\quad$ **TRAIN-ONE-EPOCH**$(\mathcal{D}_{custom}, \Theta, \Theta_{train}, \eta=5 \times 10^{-5})$
21. 
22. **// Phase 4: Fine-tune entire network (Epochs 81-100)**
23. **UNFREEZE-PARAMETERS**$(\Theta_{backbone})$ **// Optional: can keep frozen**
24. $\Theta_{train} \leftarrow \Theta$ **// All parameters**
25. 
26. **for** $epoch = \lfloor 0.8E \rfloor + 1$ **to** $E$ **do**
27. $\quad$ **TRAIN-ONE-EPOCH**$(\mathcal{D}_{custom}, \Theta, \Theta_{train}, \eta=10^{-5})$
28. 
29. $\Theta^{*} \leftarrow \Theta$
30. **return** $\Theta^{*}$

---

### **ALGORITHM S3.5: DATA-AUGMENTATION-PIPELINE**

**Input:** Image $I \in \mathbb{R}^{H \times W \times 3}$, annotations $Y = \{Y^{det}, Y^{seg}, Y^{plate}, Y^{ocr}\}$

**Output:** Augmented image $I'$, augmented annotations $Y'$

1. **// Apply geometric transformations**
2. $p \leftarrow$ **UNIFORM**$(0, 1)$
3. 
4. **// Horizontal flip (50% probability)**
5. **if** $p < 0.5$ **then**
6. $\quad$ $I \leftarrow$ **HORIZONTAL-FLIP**$(I)$
7. $\quad$ $Y^{det} \leftarrow$ **FLIP-BBOXES-HORIZONTAL**$(Y^{det}, W)$
8. $\quad$ $Y^{seg} \leftarrow$ **HORIZONTAL-FLIP**$(Y^{seg})$
9. $\quad$ $Y^{plate} \leftarrow$ **FLIP-BBOXES-HORIZONTAL**$(Y^{plate}, W)$
10. 
11. **// Random scale (0.8 to 1.2)**
12. $scale \leftarrow$ **UNIFORM**$(0.8, 1.2)$
13. $I \leftarrow$ **RESIZE**$(I, scale)$
14. $Y^{det} \leftarrow$ **SCALE-BBOXES**$(Y^{det}, scale)$
15. $Y^{seg} \leftarrow$ **RESIZE**$(Y^{seg}, scale)$
16. $Y^{plate} \leftarrow$ **SCALE-BBOXES**$(Y^{plate}, scale)$
17. 
18. **// Random crop to original size**
19. $x_{crop}, y_{crop} \leftarrow$ **RANDOM-CROP-POSITION**$(I, H, W)$
20. $I \leftarrow$ **CROP**$(I, x_{crop}, y_{crop}, H, W)$
21. $Y^{det} \leftarrow$ **ADJUST-BBOXES-AFTER-CROP**$(Y^{det}, x_{crop}, y_{crop})$
22. $Y^{seg} \leftarrow$ **CROP**$(Y^{seg}, x_{crop}, y_{crop}, H, W)$
23. $Y^{plate} \leftarrow$ **ADJUST-BBOXES-AFTER-CROP**$(Y^{plate}, x_{crop}, y_{crop})$
24. 
25. **// Color jittering**
26. $brightness \leftarrow$ **UNIFORM**$(0.8, 1.2)$
27. $contrast \leftarrow$ **UNIFORM**$(0.8, 1.2)$
28. $saturation \leftarrow$ **UNIFORM**$(0.8, 1.2)$
29. $I \leftarrow$ **ADJUST-COLOR**$(I, brightness, contrast, saturation)$
30. 
31. **// Gaussian noise (10% probability)**
32. $p \leftarrow$ **UNIFORM**$(0, 1)$
33. **if** $p < 0.1$ **then**
34. $\quad$ $noise \leftarrow$ **GAUSSIAN**$(0, \sigma=0.01, shape=(H, W, 3))$
35. $\quad$ $I \leftarrow$ **CLIP**$(I + noise, 0, 1)$
36. 
37. **// Gaussian blur (10% probability)**
38. $p \leftarrow$ **UNIFORM**$(0, 1)$
39. **if** $p < 0.1$ **then**
40. $\quad$ $\sigma_{blur} \leftarrow$ **UNIFORM**$(0.5, 2.0)$
41. $\quad$ $I \leftarrow$ **GAUSSIAN-BLUR**$(I, \sigma_{blur})$
42. 
43. $I' \leftarrow I$
44. $Y' \leftarrow Y$
45. 
46. **return** $I', Y'$

---

### **ALGORITHM S3.6: COMPUTE-MOTA** (Multi-Object Tracking Accuracy)

**Input:** Predicted tracks $\mathcal{T}_{pred}$, ground truth tracks $\mathcal{T}_{gt}$, sequence of frames

**Output:** MOTA score $\in [-\infty, 1]$ (higher is better)

1. $FN \leftarrow 0$ **// False negatives (missed detections)**
2. $FP \leftarrow 0$ **// False positives**
3. $IDSW \leftarrow 0$ **// ID switches**
4. $GT \leftarrow 0$ **// Total ground truth objects**
5. 
6. **for each** frame $t$ **in** sequence **do**
7. $\quad$ $tracks_t^{pred} \leftarrow \mathcal{T}_{pred}[t]$
8. $\quad$ $tracks_t^{gt} \leftarrow \mathcal{T}_{gt}[t]$
9. $\quad$
10. $\quad$ $GT \leftarrow GT + |tracks_t^{gt}|$
11. $\quad$
12. $\quad$ **// Find matches based on IoU threshold (0.5)**
13. $\quad$ $matches \leftarrow$ **MATCH-BY-IOU**$(tracks_t^{pred}, tracks_t^{gt}, threshold=0.5)$
14. $\quad$
15. $\quad$ **// Count false negatives**
16. $\quad$ $FN \leftarrow FN + (|tracks_t^{gt}| - |matches|)$
17. $\quad$
18. $\quad$ **// Count false positives**
19. $\quad$ $FP \leftarrow FP + (|tracks_t^{pred}| - |matches|)$
20. $\quad$
21. $\quad$ **// Count ID switches**
22. $\quad$ **for each** $(pred_i, gt_j) \in matches$ **do**
23. $\quad\quad$ **if** $t > 1$ **and** $gt_j$ **was matched to different** $pred_k$ **in frame** $t-1$ **then**
24. $\quad\quad\quad$ $IDSW \leftarrow IDSW + 1$
25. 
26. **// Compute MOTA**
27. $MOTA \leftarrow 1 - \frac{FN + FP + IDSW}{GT}$
28. 
29. **return** $MOTA$

---

## **SUMMARY OF THREE-STAGE TRAINING**

### **Stage 1: Initialization**
- Load pre-trained ImageNet/COCO backbone
- Initialize transformer encoder and all decoders from scratch
- **Time:** ~1 hour (just loading/initialization)
- **GPU:** Not needed

### **Stage 2: Task-Specific Pre-training**
- Fine-tune each decoder independently on public datasets
- Detection: COCO (50-100K images), 20-30 epochs, ~2-3 days
- Segmentation: Cityscapes (5K images), 50-100 epochs, ~1-2 days
- Plate: CCPD (250K images), 20-30 epochs, ~2-3 days
- OCR: CCPD + Synthetic (300K images), 30-50 epochs, ~2-3 days
- Tracking: MOT17/BDD100K (10K videos), 20-30 epochs, ~3-5 days
- **Total Time:** ~2 weeks
- **GPU:** 1-2 × A100 or V100

### **Stage 3: Joint Multi-Task Training**
- Train entire architecture end-to-end on custom dataset
- Custom data: 10K images, 50-100 epochs, ~3-5 days
- **Time:** ~1 week
- **GPU:** 2-4 × A100 or V100

### **Total Training Pipeline:**
- **Time:** ~3-4 weeks
- **Compute Cost:** ~$5,000-15,000 (cloud GPU rental)
- **Final Performance:** 80-90% of fully supervised single-task models, but unified and efficient

This three-stage approach dramatically reduces the custom data requirement from 50K+ images (if training from scratch) to just 10K images by leveraging transfer learning effectively.