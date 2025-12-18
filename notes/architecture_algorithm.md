
## **ALGORITHM 1: UNIFIED-MULTI-TASK-TRANSFORMER**

**Input:** Image $I \in \mathbb{R}^{H \times W \times 3}$, previous frame tracking state $T_{prev}$

**Output:** Detection results $\mathcal{D}$, Segmentation masks $\mathcal{S}$, Plate detections $\mathcal{P}$, OCR results $\mathcal{O}$, Tracking associations $\mathcal{T}$, Alert flags $\mathcal{A}$

1. **// Feature Extraction Stage**
2. $\mathcal{F} \leftarrow$ **CNN-BACKBONE**$(I)$ // Extract multi-scale features
3. $C_3, C_4, C_5 \leftarrow$ **EXTRACT-PYRAMID**$(\mathcal{F})$ // Get features at scales $\frac{H}{8}, \frac{H}{16}, \frac{H}{32}$
4. 
5. **// Transformer Encoding Stage**
6. $\mathcal{F}_{flat} \leftarrow$ **FLATTEN-CONCAT**$(C_3, C_4, C_5)$ // Concatenate all scales
7. $\mathcal{F}_{pos} \leftarrow$ **ADD-POSITIONAL-ENCODING**$(\mathcal{F}_{flat})$ // Add spatial position information
8. $\mathcal{M} \leftarrow$ **TRANSFORMER-ENCODER**$(\mathcal{F}_{pos}, L=6)$ // Memory: $N \times D$ tensor
9. 
10. **// Initialize Task-Specific Queries**
11. $Q_{det} \leftarrow$ **INIT-LEARNABLE-QUERIES**(100, $D$) // Detection queries
12. $Q_{seg} \leftarrow$ **INIT-LEARNABLE-QUERIES**(50, $D$) // Segmentation queries
13. $Q_{plate} \leftarrow$ **INIT-LEARNABLE-QUERIES**(50, $D$) // Plate detection queries
14. $Q_{ocr} \leftarrow$ **INIT-LEARNABLE-QUERIES**(20, $D$) // OCR queries
15. $Q_{track} \leftarrow$ **INIT-TEMPORAL-QUERIES**($T_{prev}$, $D$) // Tracking queries with temporal info
16. 
17. **// Hierarchical Parallel Decoding Stage**
18. $\mathcal{D}_{feat}, \mathcal{D} \leftarrow$ **DETECTION-DECODER**$(Q_{det}, \mathcal{M}, \text{null})$ // Stage 1: Independent
19. $\mathcal{S}_{feat}, \mathcal{S} \leftarrow$ **SEGMENTATION-DECODER**$(Q_{seg}, \mathcal{M}, \mathcal{D}_{feat})$ // Stage 1: Uses detection
20. 
21. $\mathcal{P}_{feat}, \mathcal{P} \leftarrow$ **PLATE-DECODER**$(Q_{plate}, \mathcal{M}, \mathcal{D}_{feat})$ // Stage 2: Uses detection
22. 
23. $\mathcal{O} \leftarrow$ **OCR-DECODER**$(Q_{ocr}, \mathcal{M}, \mathcal{P}_{feat})$ // Stage 3: Uses plate features
24. 
25. $\mathcal{T} \leftarrow$ **TRACKING-DECODER**$(Q_{track}, \mathcal{M}, \mathcal{D}_{feat}, \mathcal{S}_{feat}, T_{prev})$ // Stage 4: Uses all
26. 
27. **// Alert Generation Stage**
28. $\mathcal{A} \leftarrow$ **GENERATE-ALERTS**$(\mathcal{T}, \mathcal{D}, \mathcal{S})$ // Apply alert logic
29. 
30. **return** $\mathcal{D}, \mathcal{S}, \mathcal{P}, \mathcal{O}, \mathcal{T}, \mathcal{A}$

---

## **ALGORITHM 2: TRANSFORMER-ENCODER**

**Input:** Feature tensor $\mathcal{F} \in \mathbb{R}^{N \times D}$, number of layers $L$

**Output:** Encoded memory $\mathcal{M} \in \mathbb{R}^{N \times D}$

1. $\mathcal{M} \leftarrow \mathcal{F}$ // Initialize with input features
2. **for** $\ell = 1$ **to** $L$ **do**
3. $\quad$ **// Multi-Head Self-Attention**
4. $\quad$ $\mathcal{M}' \leftarrow$ **MULTI-HEAD-SELF-ATTENTION**$(\mathcal{M}, \mathcal{M}, \mathcal{M})$ 
5. $\quad$ $\mathcal{M} \leftarrow$ **LAYER-NORM**$(\mathcal{M} + \mathcal{M}')$ // Residual connection + normalization
6. $\quad$
7. $\quad$ **// Feed-Forward Network**
8. $\quad$ $\mathcal{M}' \leftarrow$ **FFN**$(\mathcal{M})$ // Two-layer MLP with ReLU
9. $\quad$ $\mathcal{M} \leftarrow$ **LAYER-NORM**$(\mathcal{M} + \mathcal{M}')$ // Residual connection + normalization
10. **return** $\mathcal{M}$

---

## **ALGORITHM 3: DETECTION-DECODER**

**Input:** Detection queries $Q_{det} \in \mathbb{R}^{100 \times D}$, encoder memory $\mathcal{M} \in \mathbb{R}^{N \times D}$, context features $\mathcal{C}$ (null for detection)

**Output:** Detection features $\mathcal{D}_{feat}$, Detection results $\mathcal{D} = \{bbox_i, class_i, color_i, type_i\}_{i=1}^{100}$

1. $Q \leftarrow Q_{det}$ // Initialize queries
2. **for** $\ell = 1$ **to** 6 **do** // 6 decoder layers
3. $\quad$ **// Self-Attention: Queries attend to each other**
4. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-SELF-ATTENTION**$(Q, Q, Q)$
5. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
6. $\quad$
7. $\quad$ **// Cross-Attention: Queries attend to encoder memory**
8. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-CROSS-ATTENTION**$(Q, \mathcal{M}, \mathcal{M})$
9. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
10. $\quad$
11. $\quad$ **// Feed-Forward Network**
12. $\quad$ $Q' \leftarrow$ **FFN**$(Q)$
13. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
14. 
15. $\mathcal{D}_{feat} \leftarrow Q$ // Store features for other decoders
16. 
17. **// Task-Specific Output Heads**
18. **for** $i = 1$ **to** 100 **do**
19. $\quad$ $bbox_i \leftarrow$ **BBOX-REGRESSION-HEAD**$(Q[i])$ // Predict $(x, y, w, h)$
20. $\quad$ $class_i \leftarrow$ **CLASSIFICATION-HEAD**$(Q[i])$ // Predict class $\in$ {car, scooty, bike, bus, truck, auto}
21. $\quad$ $color_i \leftarrow$ **COLOR-HEAD**$(Q[i])$ // Predict vehicle color
22. $\quad$ $type_i \leftarrow$ **TYPE-HEAD**$(Q[i])$ // Predict vehicle type/model
23. 
24. **return** $\mathcal{D}_{feat}$, $\mathcal{D}$

---

## **ALGORITHM 4: SEGMENTATION-DECODER**

**Input:** Segmentation queries $Q_{seg} \in \mathbb{R}^{50 \times D}$, encoder memory $\mathcal{M}$, detection features $\mathcal{D}_{feat}$

**Output:** Segmentation features $\mathcal{S}_{feat}$, Segmentation masks $\mathcal{S} = \{mask_{driveway}, mask_{footpath}\}$

1. $Q \leftarrow Q_{seg}$
2. **for** $\ell = 1$ **to** 6 **do**
3. $\quad$ **// Self-Attention**
4. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-SELF-ATTENTION**$(Q, Q, Q)$
5. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
6. $\quad$
7. $\quad$ **// Cross-Attention with Encoder Memory**
8. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-CROSS-ATTENTION**$(Q, \mathcal{M}, \mathcal{M})$
9. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
10. $\quad$
11. $\quad$ **// Inter-Decoder Cross-Attention with Detection Features**
12. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-CROSS-ATTENTION**$(Q, \mathcal{D}_{feat}, \mathcal{D}_{feat})$
13. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
14. $\quad$
15. $\quad$ **// Feed-Forward Network**
16. $\quad$ $Q' \leftarrow$ **FFN**$(Q)$
17. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
18. 
19. $\mathcal{S}_{feat} \leftarrow Q$
20. 
21. **// Generate Segmentation Masks**
22. $mask_{driveway} \leftarrow$ **MASK-HEAD**$(Q, \mathcal{M})$ // Pixel-wise driveway segmentation
23. $mask_{footpath} \leftarrow$ **MASK-HEAD**$(Q, \mathcal{M})$ // Pixel-wise footpath segmentation
24. 
25. **return** $\mathcal{S}_{feat}$, $\mathcal{S}$

---

## **ALGORITHM 5: PLATE-DECODER**

**Input:** Plate queries $Q_{plate} \in \mathbb{R}^{50 \times D}$, encoder memory $\mathcal{M}$, detection features $\mathcal{D}_{feat}$

**Output:** Plate features $\mathcal{P}_{feat}$, Plate detections $\mathcal{P} = \{plate\_bbox_i\}_{i=1}^{50}$

1. $Q \leftarrow Q_{plate}$
2. **for** $\ell = 1$ **to** 6 **do**
3. $\quad$ **// Self-Attention**
4. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-SELF-ATTENTION**$(Q, Q, Q)$
5. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
6. $\quad$
7. $\quad$ **// Cross-Attention with Encoder Memory**
8. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-CROSS-ATTENTION**$(Q, \mathcal{M}, \mathcal{M})$
9. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
10. $\quad$
11. $\quad$ **// Inter-Decoder Cross-Attention: Attend to Detection Features**
12. $\quad$ **// This constrains plate search to vehicle regions**
13. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-CROSS-ATTENTION**$(Q, \mathcal{D}_{feat}, \mathcal{D}_{feat})$
14. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
15. $\quad$
16. $\quad$ **// Feed-Forward Network**
17. $\quad$ $Q' \leftarrow$ **FFN**$(Q)$
18. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
19. 
20. $\mathcal{P}_{feat} \leftarrow Q$
21. 
22. **// Plate Bounding Box Regression**
23. **for** $i = 1$ **to** 50 **do**
24. $\quad$ $plate\_bbox_i \leftarrow$ **BBOX-REGRESSION-HEAD**$(Q[i])$ // Predict plate location
25. 
26. **return** $\mathcal{P}_{feat}$, $\mathcal{P}$

---

## **ALGORITHM 6: OCR-DECODER**

**Input:** OCR queries $Q_{ocr} \in \mathbb{R}^{20 \times D}$, encoder memory $\mathcal{M}$, plate features $\mathcal{P}_{feat}$

**Output:** OCR results $\mathcal{O} = \{text_i, confidence_i\}_{i=1}^{50}$

1. $Q \leftarrow Q_{ocr}$
2. **for** $\ell = 1$ **to** 6 **do**
3. $\quad$ **// Self-Attention**
4. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-SELF-ATTENTION**$(Q, Q, Q)$
5. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
6. $\quad$
7. $\quad$ **// Cross-Attention with Encoder Memory**
8. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-CROSS-ATTENTION**$(Q, \mathcal{M}, \mathcal{M})$
9. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
10. $\quad$
11. $\quad$ **// CRITICAL: Inter-Decoder Cross-Attention with Plate Features**
12. $\quad$ **// OCR queries attend to exact plate locations and orientations**
13. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-CROSS-ATTENTION**$(Q, \mathcal{P}_{feat}, \mathcal{P}_{feat})$
14. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
15. $\quad$
16. $\quad$ **// Feed-Forward Network**
17. $\quad$ $Q' \leftarrow$ **FFN**$(Q)$
18. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
19. 
20. **// Character Recognition using CTC (Connectionist Temporal Classification)**
21. **for each** detected plate $p \in \mathcal{P}$ **do**
22. $\quad$ $text_p \leftarrow$ **CTC-DECODE**$(Q, p)$ // Decode character sequence
23. $\quad$ $confidence_p \leftarrow$ **CONFIDENCE-SCORE**$(Q, p)$ // Per-character confidence
24. $\quad$ $\mathcal{O} \leftarrow \mathcal{O} \cup \{(text_p, confidence_p)\}$
25. 
26. **return** $\mathcal{O}$

---

## **ALGORITHM 7: TRACKING-DECODER**

**Input:** Tracking queries $Q_{track}$, encoder memory $\mathcal{M}$, detection features $\mathcal{D}_{feat}$, segmentation features $\mathcal{S}_{feat}$, previous tracking state $T_{prev}$

**Output:** Tracking associations $\mathcal{T} = \{track\_id_i, trajectory_i, stopped\_time_i\}$

1. **// Initialize tracking queries with temporal embeddings**
2. $Q \leftarrow$ **AUGMENT-WITH-TEMPORAL**$(Q_{track}, T_{prev})$ // Add frame-to-frame motion priors
3. 
4. **for** $\ell = 1$ **to** 6 **do**
5. $\quad$ **// Self-Attention: Track queries coordinate**
6. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-SELF-ATTENTION**$(Q, Q, Q)$
7. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
8. $\quad$
9. $\quad$ **// Cross-Attention with Encoder Memory**
10. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-CROSS-ATTENTION**$(Q, \mathcal{M}, \mathcal{M})$
11. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
12. $\quad$
13. $\quad$ **// Inter-Decoder Cross-Attention #1: Attend to Detection Features**
14. $\quad$ **// Learn: Vehicle locations, classes for association**
15. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-CROSS-ATTENTION**$(Q, \mathcal{D}_{feat}, \mathcal{D}_{feat})$
16. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
17. $\quad$
18. $\quad$ **// Inter-Decoder Cross-Attention #2: Attend to Segmentation Features**
19. $\quad$ **// Learn: Overlap with driveway/footpath for alert logic**
20. $\quad$ $Q' \leftarrow$ **MULTI-HEAD-CROSS-ATTENTION**$(Q, \mathcal{S}_{feat}, \mathcal{S}_{feat})$
21. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
22. $\quad$
23. $\quad$ **// Feed-Forward Network**
24. $\quad$ $Q' \leftarrow$ **FFN**$(Q)$
25. $\quad$ $Q \leftarrow$ **LAYER-NORM**$(Q + Q')$
26. 
27. **// Track Association and Trajectory Prediction**
28. $\mathcal{T} \leftarrow$ **ASSOCIATE-TRACKS**$(Q, T_{prev}, \mathcal{D})$ // Hungarian matching or learned association
29. 
30. **// Update Stopped Time for Each Track**
31. **for each** track $t \in \mathcal{T}$ **do**
32. $\quad$ **if** **IS-STATIONARY**$(t.trajectory)$ **then**
33. $\quad\quad$ $t.stopped\_time \leftarrow t.stopped\_time + \Delta t$ // Increment stopped duration
34. $\quad$ **else**
35. $\quad\quad$ $t.stopped\_time \leftarrow 0$ // Reset if moving
36. 
37. **return** $\mathcal{T}$

---

## **ALGORITHM 8: GENERATE-ALERTS**

**Input:** Tracking results $\mathcal{T}$, Detection results $\mathcal{D}$, Segmentation masks $\mathcal{S}$

**Output:** Alert flags $\mathcal{A} = \{alert_i \in \{\text{RED}, \text{NORMAL}\}\}$

1. Initialize $\mathcal{A} \leftarrow \emptyset$
2. 
3. **// Extract driveway mask from segmentation**
4. $mask_{driveway} \leftarrow \mathcal{S}.mask_{driveway}$
5. 
6. **for each** track $t \in \mathcal{T}$ **do**
7. $\quad$ **// Retrieve corresponding vehicle bounding box**
8. $\quad$ $bbox_t \leftarrow$ **GET-BBOX-FOR-TRACK**$(t, \mathcal{D})$
9. $\quad$
10. $\quad$ **// Compute Intersection over Union with driveway**
11. $\quad$ $overlap \leftarrow$ **COMPUTE-IOU**$(bbox_t, mask_{driveway})$
12. $\quad$
13. $\quad$ **// Apply Alert Condition**
14. $\quad$ **if** $(t.stopped\_time > 120 \text{ seconds})$ **and** $(overlap > 0.3)$ **then**
15. $\quad\quad$ $alert_t \leftarrow \text{RED}$ // Mark as violation
16. $\quad$ **else**
17. $\quad\quad$ $alert_t \leftarrow \text{NORMAL}$
18. $\quad$
19. $\quad$ $\mathcal{A} \leftarrow \mathcal{A} \cup \{(t.track\_id, alert_t)\}$
20. 
21. **return** $\mathcal{A}$

---

## **ALGORITHM 9: MULTI-HEAD-CROSS-ATTENTION**

**Input:** Queries $Q \in \mathbb{R}^{n_q \times D}$, Keys $K \in \mathbb{R}^{n_k \times D}$, Values $V \in \mathbb{R}^{n_k \times D}$, number of heads $h$

**Output:** Attended features $O \in \mathbb{R}^{n_q \times D}$

1. $d_k \leftarrow D / h$ // Dimension per head
2. 
3. **// Project to multiple heads**
4. **for** $i = 1$ **to** $h$ **do**
5. $\quad$ $Q_i \leftarrow Q \cdot W_i^Q$ where $W_i^Q \in \mathbb{R}^{D \times d_k}$
6. $\quad$ $K_i \leftarrow K \cdot W_i^K$ where $W_i^K \in \mathbb{R}^{D \times d_k}$
7. $\quad$ $V_i \leftarrow V \cdot W_i^V$ where $W_i^V \in \mathbb{R}^{D \times d_k}$
8. 
9. **// Compute attention for each head**
10. **for** $i = 1$ **to** $h$ **do**
11. $\quad$ $scores_i \leftarrow \frac{Q_i \cdot K_i^T}{\sqrt{d_k}}$ // Scaled dot-product
12. $\quad$ $attn\_weights_i \leftarrow$ **SOFTMAX**$(scores_i)$ // Normalize across keys
13. $\quad$ $head_i \leftarrow attn\_weights_i \cdot V_i$ // Weighted sum of values
14. 
15. **// Concatenate all heads and project**
16. $O \leftarrow$ **CONCAT**$(head_1, head_2, \ldots, head_h) \cdot W^O$ where $W^O \in \mathbb{R}^{D \times D}$
17. 
18. **return** $O$

---

## **ALGORITHM 10: TRAINING-PROCEDURE**

**Input:** Training dataset $\mathcal{D}_{train} = \{(I_j, Y_j^{det}, Y_j^{seg}, Y_j^{plate}, Y_j^{ocr}, Y_j^{track})\}_{j=1}^{N}$, learning rate $\eta$, loss weights $\{\lambda_1, \lambda_2, \lambda_3, \lambda_4, \lambda_5\}$

**Output:** Trained model parameters $\Theta$

1. **// Initialize all model parameters**
2. $\Theta \leftarrow$ **INITIALIZE-PARAMETERS**() // Xavier/He initialization
3. 
4. **for** epoch $= 1$ **to** $E$ **do**
5. $\quad$ **for each** batch $(I, Y)$ **in** $\mathcal{D}_{train}$ **do**
6. $\quad\quad$ **// Forward pass**
7. $\quad\quad$ $\mathcal{D}, \mathcal{S}, \mathcal{P}, \mathcal{O}, \mathcal{T}, \mathcal{A} \leftarrow$ **UNIFIED-MULTI-TASK-TRANSFORMER**$(I, T_{prev})$
8. $\quad\quad$
9. $\quad\quad$ **// Compute individual task losses**
10. $\quad\quad$ $\mathcal{L}_{det} \leftarrow$ **DETECTION-LOSS**$(\mathcal{D}, Y^{det})$ // Focal loss + GIoU loss
11. $\quad\quad$ $\mathcal{L}_{seg} \leftarrow$ **SEGMENTATION-LOSS**$(\mathcal{S}, Y^{seg})$ // Cross-entropy + Dice loss
12. $\quad\quad$ $\mathcal{L}_{plate} \leftarrow$ **PLATE-LOSS**$(\mathcal{P}, Y^{plate})$ // Focal loss + GIoU loss
13. $\quad\quad$ $\mathcal{L}_{ocr} \leftarrow$ **CTC-LOSS**$(\mathcal{O}, Y^{ocr})$ // Connectionist Temporal Classification loss
14. $\quad\quad$ $\mathcal{L}_{track} \leftarrow$ **TRACKING-LOSS**$(\mathcal{T}, Y^{track})$ // Association loss + trajectory loss
15. $\quad\quad$
16. $\quad\quad$ **// Compute total multi-task loss**
17. $\quad\quad$ $\mathcal{L}_{total} \leftarrow \lambda_1 \mathcal{L}_{det} + \lambda_2 \mathcal{L}_{seg} + \lambda_3 \mathcal{L}_{plate} + \lambda_4 \mathcal{L}_{ocr} + \lambda_5 \mathcal{L}_{track}$
18. $\quad\quad$
19. $\quad\quad$ **// Backward pass and parameter update**
20. $\quad\quad$ $\nabla_\Theta \mathcal{L}_{total} \leftarrow$ **BACKPROPAGATE**$(\mathcal{L}_{total})$
21. $\quad\quad$ $\Theta \leftarrow \Theta - \eta \cdot \nabla_\Theta \mathcal{L}_{total}$ // Gradient descent (or Adam optimizer)
22. 
23. **return** $\Theta$

---

## **Key Algorithmic Insights:**

1. **Hierarchical Information Flow:** Detection and Segmentation run first (independent), then Plate Detection uses detection features, OCR uses plate features, and Tracking uses both detection and segmentation.

2. **Inter-Decoder Cross-Attention:** Unlike standard transformers where decoders only attend to encoder memory, this architecture has strategic cross-attention connections between decoders at specific stages (lines 12-14 in Algorithms 4, 5, 6, 7).

3. **Three Types of Attention per Decoder:**
   - Self-attention: Queries coordinate with each other
   - Cross-attention with memory: Extract image features
   - Cross-attention with other decoders: Share task-specific knowledge

4. **Alert Logic:** Post-processing rule (Algorithm 8) that doesn't require learning—it's a deterministic computation based on tracking duration and spatial overlap.

5. **End-to-End Differentiability:** All components from CNN backbone through all decoders to output heads are differentiable, allowing joint optimization via backpropagation (Algorithm 10).

This architecture achieves true parallelism where possible (detection + segmentation) while respecting logical dependencies where necessary (plate→OCR, detection+seg→tracking).