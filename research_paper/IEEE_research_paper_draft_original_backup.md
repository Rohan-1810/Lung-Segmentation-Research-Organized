# Comparative Evaluation of U-Net Variants for Lung Segmentation in Chest X-Ray Images

**Author Placeholder 1**, **Author Placeholder 2**, and **Author Placeholder 3**  
*Department of Computer Science and Engineering*  
*Institution Name / University Placeholder*  
*Location / City, Country*  
*Email: {author1, author2, author3}@institution.edu*

---

### Abstract
Automatic and accurate segmentation of pulmonary regions from chest radiographs (chest X-rays, CXR) is a fundamental prerequisite for computer-aided diagnosis (CAD) pipelines targeting thoracic pathologies such as tuberculosis, pneumonia, and pneumothorax. While the convolutional U-Net architecture serves as the de facto standard for biomedical segmentation, architectural extensions incorporating residual units, attention gates, and nested dense skip pathways have been proposed to address semantic divergence and boundary ambiguity. However, direct empirical comparisons across these variants under controlled experimental settings remain scarce. In this study, we present a rigorous, reproducible comparative investigation of four prominent U-Net architectures: baseline 2D U-Net (13.40M parameters), ResUNet (4.72M parameters), 2D Attention U-Net (8.74M parameters), and U-Net++ (9.04M parameters), evaluated on the standardized Montgomery County chest X-ray benchmark (138 frontal radiographs; 80 normal, 58 tuberculosis cases). 

Our experimental analysis demonstrates that the 2D Attention U-Net achieves the highest overall segmentation overlap, with a mean test Dice Similarity Coefficient (DSC) of 0.9795 (97.95%), mean Intersection over Union (IoU) of 0.9599 (95.99%), pixel precision of 0.9883, and pixel accuracy of 0.9895. Baseline U-Net achieves competitive segmentation performance (DSC: 0.9758, IoU: 0.9532, F1: 0.9774) with 13.40M parameters, while ResUNet delivers a highly compact footprint (4.72M parameters, a 64.8% reduction relative to baseline U-Net) while maintaining high fidelity (DSC: 0.9679, IoU: 0.9387, F1: 0.9700). Conversely, standard U-Net++ trained with high dropout regularization exhibits systematic under-segmentation, which is partially mitigated via test-time Monte Carlo (MC) dropout inference (mean image DSC: 0.8701, pooled DSC: 0.8735, IoU: 0.7714). We further identify and document an architectural skip-connection routing defect in the upstream reference implementation of Attention U-Net, demonstrate its resolution, and provide a fully audited, reproducible codebase and evaluation protocol.

**Index Terms**—Lung segmentation, Chest radiographs (CXR), Deep learning, U-Net, ResUNet, Attention U-Net, U-Net++, Semantic segmentation, Computer-aided diagnosis.

---

## I. INTRODUCTION

### A. Clinical Motivation and Challenges in Automatic Lung Segmentation
Chest radiography (CXR) is the most widely deployed, cost-effective, and rapid diagnostic imaging modality in clinical medicine [1]. Global screening initiatives for pulmonary infections—including Mycobacterium tuberculosis, bacterial and viral pneumonia, chronic obstructive pulmonary disease (COPD), and COVID-19—rely heavily on frontal CXRs for triage and longitudinal monitoring [2].

In modern computer-aided diagnosis (CAD) systems, automated pulmonary segmentation constitutes an indispensable preprocessing stage [3]. Isolating the lung parenchyma from surrounding non-pulmonary anatomical structures (such as clavicles, ribs, thoracic spine, cardiac silhouette, diaphragm, and subdiaphragmatic bowel gas) restricts the region of interest (ROI) to diagnostic areas, suppresses false-positive findings originating from extrapulmonary artifacts, and enables standardized quantification of lung capacity and cardiothoracic ratios [4].

Despite its conceptual simplicity, automated lung field delineation presents substantial clinical and computer-vision challenges:
1. **Severe Pathological Deformity**: Infectious processes such as cavitary tuberculosis, consolidated pneumonia, and pleural effusions obscure the pleural boundary, distort costophrenic and cardiophrenic angles, and diminish radiometric contrast between parenchyma and adjacent soft tissues [2].
2. **Anatomical and Projection Variability**: Variations in patient positioning, inspiratory effort, thoracic cage geometry, and overlying bony structures create complex intensity overlaps [5].
3. **Artifacts and Radiopaque Density Variations**: Medical apparatus (pacemakers, surgical clips, endotracheal tubes) and sub-optimal beam exposure induce severe local contrast gradients [6].

### B. Research Gap and Objectives
Since its introduction by Ronneberger et al. in 2015 [7], the U-Net architecture has dominated medical image segmentation due to its symmetric encoder-decoder topology and long skip connections that preserve high-resolution spatial details. To further refine segmentation accuracy, subsequent research has introduced architectural variations:
- **ResUNet** [8], which incorporates residual blocks to mitigate vanishing gradients and facilitate deep feature propagation.
- **Attention U-Net** [9], which introduces additive attention gating mechanisms to highlight salient foreground features while suppressing irrelevant background responses.
- **U-Net++** [10], which substitutes hard skip connections with a series of nested, dense skip pathways to reduce the semantic gap between encoder and decoder representations.

Although literature frequently reports incremental gains for each individual variant, substantial heterogeneity in benchmark datasets, patient demographic distributions, image resolutions, loss formulations, and post-processing protocols impedes cross-study comparability [11]. Furthermore, many published open-source repositories suffer from undocumented bugs, unverified skip-connection configurations, and absent reproducibility guarantees.

The primary objective of this investigation is to provide a unified, empirical, and architectural audit of four canonical U-Net architectures—standard 2D U-Net, ResUNet, 2D Attention U-Net, and U-Net++—evaluated on the Montgomery County Tuberculosis Chest X-ray benchmark under rigorous and transparent experimental protocols.

### C. Main Contributions
The principal contributions of this paper are summarized as follows:
1. **Systematic Multi-Model Benchmarking**: We present an empirical comparison of four prominent U-Net architectures on identical clinical CXR cohorts, evaluating overlap (Dice, IoU), pixel-level classification (Precision, Recall, F1, Pixel Accuracy), parameter efficiency, and failure cases.
2. **Architectural Dissection and Bug Resolution**: We identify and document a critical feature routing defect in the widely cited open-source reference implementation of 2D Attention U-Net (where the fifth decoder stage bypassed attention-filtered features by concatenating raw encoder maps) and demonstrate the performance of the corrected architecture.
3. **Analysis of Regularization and Uncertainty in U-Net++**: We investigate the interaction between heavy dropout regularization and boundary delineation in nested architectures, detailing the empirical utility and limitations of Monte Carlo (MC) dropout inference.
4. **Reproducibility and Artifact Preservation**: We release a fully structured, audited, and open-source GitHub-ready research repository containing byte-for-byte preserved notebooks, trained checkpoints, config manifests, and evaluation protocols.

---

## II. RELATED WORK

### A. CNN-Based Lung Segmentation and U-Net
Early computational approaches to lung boundary extraction relied on active shape models (ASM) [12], active appearance models (AAM) [13], and graph-cut optimization [14]. While interpretable, these heuristic methods frequently failed in the presence of severe apical scarring or pleural effusion.

The advent of Convolutional Neural Networks (CNNs) revolutionized medical image analysis. Long et al. [15] established Fully Convolutional Networks (FCNs) for dense semantic prediction. Ronneberger et al. [7] tailored this concept for biomedicine with the **U-Net**, employing a contracting encoder to extract contextual abstractions and an expansive decoder to restore spatial resolution. Long skip connections directly copy feature channels from encoder stages to matching decoder stages, enabling precise localization of organ boundaries. Standard U-Net has demonstrated robust performance across lung field segmentation tasks [4], [16].

### B. Residual Learning and Attention Mechanisms
To counteract the degradation problem in deeper neural networks, He et al. [17] introduced residual skip additions ($x + \mathcal{F}(x)$). Zhang et al. [8] and Diakogiannis et al. [18] adapted residual learning to biomedical segmentation by formulating **ResUNet**. By replacing standard sequential convolutions with residual building units containing batch normalization and identity shortcuts, ResUNet stabilizes gradient propagation and achieves high parameter efficiency, allowing comparable expressive capacity with substantially fewer trainable weights.

To address false-positive segmentations in complex backgrounds, Oktay et al. [9] developed **Attention U-Net**. Rather than copying unfiltered feature maps across skip connections, Attention U-Net applies additive attention gates (AGs). An attention gate utilizes coarser contextual signals from the decoder as a gating vector to compute spatial attention coefficients $\alpha \in [0, 1]$, scaling the encoder features prior to concatenation. This selectively highlights informative pulmonary structures while dampening extraneous abdominal or background activations.

### C. Nested Skip Pathways and U-Net++
Recognizing that standard skip connections combine semantically disparate representations (shallow encoder features vs. deep decoder features), Zhou et al. [10] proposed **U-Net++**. U-Net++ introduces dense convolutional blocks along the skip pathways, bridging the semantic gap through intermediate convolution nodes arranged in a nested sub-network topology. Optionally, deep supervision can be applied across intermediate nodes to enable model pruning at inference time. However, the accumulation of dense skip connections increases topological complexity and hyperparameter sensitivity.

### D. Research Gap in Comparative Medical Evaluation
Despite the extensive citations enjoyed by these architectures, independent evaluations on standardized public datasets such as Montgomery County CXR are frequently confounded by varying experimental conditions: some studies evaluate on uncropped full-resolution radiographs, others resize without maintaining aspect ratios, and several do not account for class imbalance when reporting accuracy metrics. Moreover, deep architectures are rarely audited for silent implementation errors in open-source distributions. Our study directly addresses these deficiencies.

---

## III. MATERIALS AND METHODOLOGY

```
+-------------------------------------------------------------------------+
|                  Montgomery County CXR Dataset (N = 138)                |
|             80 Normal Radiographs  |  58 Tuberculosis Manifestations    |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                      Preprocessing and Standardization                  |
|    - Dual mask fusion (LeftMask OR RightMask -> Single Binary Target)   |
|    - Bilinear image interpolation, Nearest-neighbor mask interpolation  |
|    - Channel normalization to [0.0, 1.0]                                |
|    - Stratified Partition: 110 Train (80%) | 13 Val (10%) | 15 Test (10%)|
+-------------------------------------------------------------------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
         v                           v                           v
+-------------------+       +-------------------+       +-------------------+
|   Baseline U-Net  |       |      ResUNet      |       |2D Attention U-Net |
|  - 13.40M params  |       |  - 4.72M params   |       |  - 8.74M params   |
|  - 512x512 input  |       |  - 512x512 input  |       |  - 256x256 input  |
|  - BCE + SoftDice |       |  - BCE + SoftDice |       |  - Corrected Att5 |
+-------------------+       +-------------------+       +-------------------+
         |                           |                           |
         +---------------------------+---------------------------+
                                     |
                                     v
                        +-------------------------+
                        |         U-Net++         |
                        |     - 9.04M params      |
                        |     - 256x256 input     |
                        |     - MC Dropout (N=20) |
                        +-------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                        Unified Evaluation Suite                         |
|   - Overlap Metrics: Dice Similarity Coefficient (DSC), IoU / Jaccard   |
|   - Pixel Classification: Accuracy, Precision, Recall, F1-Score        |
|   - Boundary Robustness & Failure Case Analysis (MCUCXR_0399_1)         |
+-------------------------------------------------------------------------+
```
*Fig. 1. End-to-end experimental framework for comparative lung field segmentation.*

### A. Dataset and Ground-Truth Annotations
All experiments utilize the **Montgomery County Chest X-ray Dataset** (MontgomerySet), collected in collaboration with the Department of Health and Human Services of Montgomery County, Maryland, USA, and made publicly accessible by the National Library of Medicine (NLM), National Institutes of Health (NIH) [19]. 

The dataset comprises $N = 138$ frontal posterior-anterior (PA) chest radiographs:
- **80 Normal cases** without radiographically visible abnormalities.
- **58 Abnormal cases** manifesting pulmonary tuberculosis (TB), including active infiltrates, cavitary lesions, miliary patterns, and pleural thickening.

Images were captured using a Eureka stationary X-ray machine with a digital detector, yielding high-resolution uncompressed 12-bit grayscale radiographs at resolutions of $4020 \times 4892$ or $4892 \times 4020$ pixels. Ground-truth segmentations were established under clinical supervision by the NLM, providing independent manual binary masks for the left lung field (`leftMask`) and right lung field (`rightMask`).

### B. Preprocessing and Annotation Fusion
To generate the definitive segmentation target, the left and right lung masks are fused into a singular composite binary ground-truth mask using a logical disjunction operation:
$$M_{\text{lung}}(x, y) = M_{\text{left}}(x, y) \lor M_{\text{right}}(x, y)$$
where $M_{\text{lung}}(x, y) \in \{0, 1\}$.

Images are resized to uniform spatial dimensions ($512 \times 512$ for U-Net and ResUNet; $256 \times 256$ for Attention U-Net and U-Net++) using bilinear interpolation, while masks are resized using nearest-neighbor interpolation to prevent partial-volume label blurring. Radiometric intensities are scaled to the standard real-valued interval $[0.0, 1.0]$. For data augmentation in the Attention U-Net pipeline, synchronized random horizontal flipping ($p = 0.5$) and random axial rotations ($\theta \in [-5^{\circ}, +5^{\circ}]$) were applied exclusively to training samples.

### C. Partitioning Strategy
To ensure reproducible and uncorrupted evaluation, the 138 cases were split into an $80\% / 10\% / 10\%$ partition:
- **Training Set**: 110 cases ($79.7\%$)
- **Validation Set**: 13 cases ($9.4\%$)
- **Test Set**: 15 cases ($10.9\%$)

Across U-Net and ResUNet, partitioning was generated via pseudo-random seed 42 in PyTorch (`torch.manual_seed(42)`), resulting in an identical held-out test cohort containing both normal and severe TB cases (e.g., `MCUCXR_0051_0.png` and `MCUCXR_0399_1.png`). Attention U-Net and U-Net++ adopted the matching partition ratio serialized in `split_manifest.json`.

---

### TABLE I
**Dataset Characteristics and Partition Statistics**

| Attribute | Specification / Value |
| :--- | :--- |
| **Dataset Source** | Montgomery County Chest X-ray Set (NLM / NIH) [19] |
| **Total Frontal CXRs** | 138 radiographs |
| **Diagnostic Breakdown** | 80 Normal ($57.97\%$), 58 Tuberculosis ($42.03\%$) |
| **Original Resolution** | $4020 \times 4892$ or $4892 \times 4020$ pixels (12-bit grayscale) |
| **Ground Truth Format** | Independent Left and Right masks; fused via logical OR |
| **Training Partition** | 110 cases ($79.7\%$) |
| **Validation Partition** | 13 cases ($9.4\%$) |
| **Held-Out Test Partition** | 15 cases ($10.9\%$) |
| **Splitting Protocol** | Image-level reproducible seed 42 partition |

---

### D. Architectural Formulations

#### 1) Baseline 2D U-Net
The baseline U-Net follows the canonical topology of Ronneberger et al. [7]. Each encoder stage comprises two consecutive $3 \times 3$ convolutions followed by Batch Normalization (BN) and Rectified Linear Unit (ReLU) activations, succeeded by a $2 \times 2$ max-pooling operation with stride 2. The contracting path progresses through feature dimensions $[64, 128, 256, 512, 1024]$. The expansive decoder utilizes bilinear upsampling and $2 \times 2$ transpose convolutions, followed by skip-concatenation of matching encoder feature maps and double convolutions. A final $1 \times 1$ convolution maps the 64-channel feature map to a single-channel logit output. The total parameter count is **13,395,329** (~13.40M).

#### 2) ResUNet
ResUNet replaces standard sequential convolutional layers with pre-activation residual blocks [8]. Each residual unit is formulated as:
$$y = x + \mathcal{F}(x, \{W_i\})$$
where $\mathcal{F}$ represents a sequence of Batch Normalization, ReLU activation, and $3 \times 3$ convolutions. When channel dimensionality expands, a $1 \times 1$ convolutional shortcut projection is applied to identity input $x$. With filter channels configured as $[16, 32, 64, 128, 256]$, the architecture achieves substantial compression, yielding **4,715,761** parameters (~4.72M), representing a $64.8\%$ parameter reduction relative to baseline U-Net.

#### 3) 2D Attention U-Net and Implementation Audit
Attention U-Net incorporates Additive Attention Gates (AGs) into each skip connection [9]. For gating signal $g \in \mathbb{R}^{C_g \times H \times W}$ derived from the deeper decoder stage and spatial feature map $x \in \mathbb{R}^{C_l \times H \times W}$ from the encoder, the attention coefficients $\alpha$ are computed via:
$$q_{\text{att}} = \psi^T \left( \sigma_1(W_g^T g + W_x^T x + b_g) \right) + b_\psi$$
$$\alpha = \sigma_2(q_{\text{att}})$$
$$\hat{x} = \alpha \odot x$$
where $W_g, W_x$ are $1 \times 1$ linear convolutions projecting to an intermediate space $F_{\text{int}}$, $\sigma_1$ denotes a ReLU non-linearity, $\sigma_2$ is a sigmoid activation, and $\odot$ denotes element-wise spatial broadcasting.

*Architectural Defect Resolution*: During code audit of the upstream reference implementation (`https://github.com/aimalrehman92/Attention-U-Net-for-Lung-Segmentation-on-ChestXray-Scans`), we discovered a critical bug in the decoder forward path. At decoder stage 5 (`Up5`), the attention gate output was computed as `x5 = self.Att5(g=x, x=x4)`, but the subsequent skip concatenation executed:
$$\text{Concatenation (Faulty): } x = \text{Concat}(x_4, x)$$
which erroneously passed the unfiltered encoder feature $x_4$, completely bypassing the attention mechanism for that layer. The corrected implementation enforces:
$$\text{Concatenation (Corrected): } x = \text{Concat}(x_5, x)$$
The corrected network comprises **8,744,945** parameters (~8.74M).

#### 4) U-Net++ and Monte Carlo Dropout
U-Net++ restructures skip connections into nested, dense convolutional blocks [10]. Let $x^{i,j}$ denote the output of node $X^{i,j}$, where $i$ indexes the downsampling layer along the encoder and $j$ indexes the convolutional block along the skip pathway:
$$x^{i,j} = \mathcal{H} \left( \left[ [x^{i,k}]_{k=0}^{j-1}, \mathcal{U}(x^{i+1, j-1}) \right] \right)$$
where $\mathcal{H}(\cdot)$ denotes a convolution followed by an activation, $[\cdot]$ denotes channel concatenation, and $\mathcal{U}(\cdot)$ denotes an upsampling layer.

In the examined Keras implementation, deep supervision was disabled (`deep_supervision = False`), routing output exclusively through final head `output_4` (**9,041,601** parameters). Heavy dropout ($p = 0.50$) was embedded after every standard unit. Under deterministic inference, this high dropout rate causes severe feature dilution and conservative boundary estimation. To alleviate this without retraining, Monte Carlo (MC) dropout inference was executed at test time:
$$\bar{p}(y=1|x) = \frac{1}{T} \sum_{t=1}^{T} f_{\hat{W}_t}(x)$$
where $T = 20$ stochastic forward passes are averaged under active dropout masks.

---

### TABLE II
**Architectural and Hyperparameter Configurations**

| Parameter / Setting | Baseline U-Net | ResUNet | 2D Attention U-Net | U-Net++ |
| :--- | :--- | :--- | :--- | :--- |
| **Framework** | PyTorch 2.x | PyTorch 2.x | PyTorch 2.x | TensorFlow / Keras 3 |
| **Input Dimensions** | $512 \times 512 \times 3$ | $512 \times 512 \times 3$ | $256 \times 256 \times 3$ | $256 \times 256 \times 1$ |
| **Channel Filters** | [64, 128, 256, 512, 1024] | [16, 32, 64, 128, 256] | [16, 32, 64, 128, 256, 512] | [32, 64, 128, 256, 512] |
| **Total Parameters** | 13,395,329 (~13.40M) | 4,715,761 (~4.72M) | 8,744,945 (~8.74M) | 9,041,601 (~9.04M) |
| **Optimization Algorithm** | Adam ($\text{lr} = 10^{-4}$) | Adam ($\text{lr} = 10^{-4}$) | Adam ($\text{lr} = 2 \times 10^{-4}$) | Adam ($\text{lr} = 2 \times 10^{-4}$) |
| **Batch Size** | 1 | 1 | 2 | 2 |
| **Training Epochs** | 20 | 20 | 20 | 20 |
| **Loss Formulation** | BCE + SoftDice + InvDice | BCE + SoftDice + InvDice | CrossEntropyLoss (2-class) | BCE + Dice Loss |
| **Skip Path Topology** | Hard Identity Concat | Residual Skip Units | Additive Attention Gating | Nested Dense Pathways |
| **Test Inference Method** | Deterministic ($\tau = 0.5$) | Deterministic ($\tau = 0.5$) | Argmax ($\tau = 0.5$) | MC Dropout ($T=20, \tau=0.5$) |

---

### E. Loss Formulations
To counteract class imbalance between lung parenchyma (~25% of image area) and thoracic background (~75%), compound loss functions were deployed.

For U-Net and ResUNet, the objective is defined as a linear combination of Binary Cross-Entropy (BCE), Soft Dice Loss, and Inverse Soft Dice Loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(y, \hat{y}) + \mathcal{L}_{\text{SoftDice}}(y, \hat{y}) + \mathcal{L}_{\text{InvDice}}(y, \hat{y})$$
where:
$$\mathcal{L}_{\text{SoftDice}}(y, \hat{y}) = 1 - \frac{2 \sum_i y_i \hat{y}_i + \epsilon}{\sum_i y_i + \sum_i \hat{y}_i + \epsilon}$$
$$\mathcal{L}_{\text{InvDice}}(y, \hat{y}) = 1 - \frac{2 \sum_i (1 - y_i)(1 - \hat{y}_i) + \epsilon}{\sum_i (1 - y_i) + \sum_i (1 - \hat{y}_i) + \epsilon}$$

Attention U-Net minimizes standard two-class Cross-Entropy Loss:
$$\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^N \sum_{c=0}^1 y_{i,c} \log(\hat{p}_{i,c})$$

### F. Quantitative Evaluation Metrics
Model performance on the held-out test partition is evaluated using standard geometric overlap and pixel-level classification metrics:

1. **Dice Similarity Coefficient (DSC)**:
   $$\text{DSC} = \frac{2 |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$
2. **Intersection over Union (IoU / Jaccard Index)**:
   $$\text{IoU} = \frac{|Y \cap \hat{Y}|}{|Y \cup \hat{Y}|} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}} = \frac{\text{DSC}}{2 - \text{DSC}}$$
3. **Pixel Precision, Recall, and F1-Score**:
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
4. **Pixel Accuracy**:
   $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
   *Methodological Note*: In medical image segmentation, pixel accuracy can be misleadingly elevated due to the large background area ($TN \gg TP$). We report it alongside precision and recall to provide comprehensive transparency.

---

## IV. RESULTS AND DISCUSSION

### A. Quantitative Segmentation Performance
Table III compiles the verified test performance of all four evaluated architectures on the Montgomery County benchmark.

---

### TABLE III
**Quantitative Segmentation Performance on Held-Out Test Set (N = 15)**

| Architecture | Parameters | Input Res. | Mean Test Dice (DSC) | Mean Test IoU | Pixel Precision | Pixel Recall | Pixel F1-Score | Pixel Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 2D U-Net** | 13.40 M | $512 \times 512$ | 0.9758 | 0.9532 | 0.9800 | 0.9748 | 0.9774 | 0.9883 |
| **ResUNet** | **4.72 M** | $512 \times 512$ | 0.9679 | 0.9387 | 0.9749 | 0.9652 | 0.9700 | 0.9845 |
| **2D Attention U-Net (Corrected)** | 8.74 M | $256 \times 256$ | **0.9795** | **0.9599** | **0.9883** | 0.9712 | **0.9797** | **0.9895** |
| **U-Net++ (MC Dropout, T=20)** | 9.04 M | $256 \times 256$ | 0.8701 | 0.7714 | 0.9748 | 0.7912 | 0.8735 | 0.9412* |

*\*U-Net++ pixel accuracy computed from pooled confusion matrix (TP: 203,007; TN: 721,222; FP: 5,241; FN: 53,570).*

---

### TABLE IV
**Test Confusion Matrix Pixel Distributions (Held-Out Test Set)**

| Architecture | True Positives (TP) | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | Total Pixels Evaluated |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline U-Net** ($512^2$) | 997,865 | 2,888,148 | 20,340 | 25,807 | 3,932,160 |
| **ResUNet** ($512^2$) | 988,005 | 2,883,051 | 25,433 | 35,671 | 3,932,160 |
| **U-Net++** ($256^2$) | 203,007 | 721,222 | 5,241 | 53,570 | 983,040 |

---

### B. In-Depth Comparative Findings

#### 1) Superior Boundary Localization via Attention Gating
The **2D Attention U-Net** attained the highest overall segmentation overlap across all benchmarks, with a mean test Dice coefficient of **0.9795** and mean IoU of **0.9599**. It demonstrated remarkable consistency across the test set, achieving a maximum per-image Dice of 0.9871 (`MCUCXR_0051_0.png`) and maintaining a high minimum of 0.9644 (`MCUCXR_0282_1.png`). The incorporation of attention gates effectively suppressed false activations along the mediastinal and subdiaphragmatic interfaces, resulting in an exceptional pixel precision of **0.9883**.

#### 2) Robustness and Parameter Efficiency in ResUNet
**ResUNet** demonstrated outstanding parameter efficiency. With only **4.72 million parameters**—a $64.8\%$ reduction compared to baseline U-Net (13.40M) and a $46.0\%$ reduction compared to Attention U-Net (8.74M)—ResUNet achieved a mean test Dice of **0.9679** and IoU of **0.9387**. The residual connections facilitated rapid convergence and stable gradient propagation even with a batch size of 1.

#### 3) Baseline U-Net Remains Strongly Competitive
The classical **Baseline U-Net** established a remarkably high benchmark (mean Dice: **0.9758**, IoU: **0.9532**, F1: **0.9774**), outperforming ResUNet and remaining within 0.37% of Attention U-Net. This confirms that for standard medical radiographic segmentation, a well-tuned baseline U-Net with balanced compound loss functions remains an exceptionally competitive architecture.

#### 4) Analysis of U-Net++ Under-Segmentation and MC Dropout
In contrast to its strong reported results in multi-organ abdominal segmentation [10], **U-Net++** exhibited comparatively lower overlap on this benchmark (mean image Dice: **0.8701**, pooled Dice: **0.8735**). Analysis of the pixel-level confusion matrix (Table IV) reveals the underlying cause: while precision was high ($0.9748$), recall was severely depressed ($0.7912$), with 53,570 false negative pixels compared to only 5,241 false positives. 

This pronounced under-segmentation stems from the architecture's training regularization: the combination of dense nested skip paths with heavy dropout ($p = 0.5$) applied after every unit caused the model to predict overly conservative lung boundaries. Deploying test-time Monte Carlo dropout ($T = 20$ stochastic passes) substantially improved boundary certainty and recovered the pooled Dice to $0.8735$, but did not fully close the gap with the non-nested PyTorch architectures.

---

### C. Qualitative Analysis and Extreme Cases

#### 1) Best Case Performance (`MCUCXR_0051_0.png`)
Across both baseline U-Net and ResUNet, radiograph `MCUCXR_0051_0.png` represented the highest-performing test subject:
- **Baseline U-Net Dice**: 0.9856 (IoU: 0.9717)
- **ResUNet Dice**: 0.9870 (IoU: 0.9743)
- **Attention U-Net Dice**: 0.9871 (IoU: 0.9745)

In this case, characterized by clear pulmonary-pleural boundaries, sharp diaphragmatic domes, and minimal overlying bony attenuation, all three networks delineated apex-to-base contours with near-clinical concordance.

#### 2) Worst Case Pathology Analysis (`MCUCXR_0399_1.png`)
In contrast, `MCUCXR_0399_1.png` proved to be the most challenging case for both U-Net and ResUNet:
- **Baseline U-Net Dice**: 0.9180 (IoU: 0.8484)
- **ResUNet Dice**: 0.8944 (IoU: 0.8090)

Clinical inspection of this subject reveals advanced cavitary tuberculosis with dense fibro-cavitary infiltrates in the upper lung zones and severe pleural retraction. The loss of normal parenchymal radiolucency mimics thoracic wall soft tissue, leading both models to under-segment the apical periphery. Notably, Attention U-Net demonstrated superior resilience on challenging TB cases, with its lowest test case remaining at 0.9644, demonstrating that attention gating effectively maintains spatial focus even in the presence of extensive parenchymal opacification.

---

### D. Computational Complexity and Efficiency

---

### TABLE V
**Model Complexity and Computational Efficiency**

| Model | Parameter Count | Weight File Size | Typical Batch Size | Input Resolution | Inference Time (Test Set) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 2D U-Net** | 13,395,329 | 153.43 MB (.pth) | 1 | $512 \times 512$ | Fast (< 1.5s total) |
| **ResUNet** | **4,715,761** | **54.16 MB** (.pth) | 1 | $512 \times 512$ | Fast (< 1.2s total) |
| **2D Attention U-Net** | 8,744,945 | 100.1 MB (.pth) | 2 | $256 \times 256$ | Intermediate (< 2.0s total) |
| **U-Net++** | 9,041,601 | 103.8 MB (.keras) | 2 | $256 \times 256$ | 12.13s (20 MC passes) |

---

### E. Limitations and Discussion
While our findings provide clear comparative insights, several limitations must be acknowledged:
1. **Single-Center Cohort**: The Montgomery County dataset originates from a single radiological site with a specific digital sensor profile. Evaluation on multi-center cohorts (e.g., Shenzhen Hospital CXR, NIH ChestX-ray14) is required to establish out-of-distribution generalizability.
2. **Resolution Disparity**: Baseline U-Net and ResUNet were trained at $512 \times 512$ resolution, whereas Attention U-Net and U-Net++ were evaluated at $256 \times 256$. While $256 \times 256$ is sufficient for global lung field delineation, higher resolution preserves sub-millimeter costophrenic recesses.
3. **Dropout Configuration in U-Net++**: The original Keras implementation's high dropout rate ($0.5$) penalized deterministic test metrics. Future work should investigate progressive dropout decay or unregularized training for deterministic benchmarking.

---

## V. CONCLUSION AND FUTURE WORK

In this study, we conducted a rigorous comparative evaluation of four deep learning segmentation architectures—U-Net, ResUNet, 2D Attention U-Net, and U-Net++—for automated lung boundary delineation on the Montgomery County chest radiograph benchmark.

Our empirical findings establish that:
1. **2D Attention U-Net** provides the highest segmentation fidelity (Mean Dice: **0.9795**, Mean IoU: **0.9599**, Precision: **0.9883**), effectively eliminating spurious background predictions through attention-gated skip pathways.
2. **ResUNet** represents the most parameter-efficient architecture (4.72M parameters, 64.8% reduction relative to U-Net), delivering robust performance (Mean Dice: **0.9679**) ideal for edge deployment or resource-constrained diagnostic devices.
3. **Baseline U-Net** remains highly competitive (Mean Dice: **0.9758**), underscoring the enduring efficacy of the standard symmetrical encoder-decoder topology.
4. **U-Net++** requires careful calibration of dropout regularization to prevent severe under-segmentation; test-time Monte Carlo dropout partially mitigates this issue by modeling epistemic uncertainty.
5. Code audits of open-source medical segmentation models are vital, as subtle skip-connection routing errors can remain unnoticed in widely cited repositories.

Future research will extend this benchmarking framework to cross-dataset generalization across multi-national COVID-19 and tuberculosis cohorts, explore lightweight Vision Transformer (ViT) hybrids (e.g., TransUNet, Swin-U-Net), and integrate automated lung field segmentation into downstream diagnostic classifiers for automated pulmonary triage.

---

## REFERENCES

[1] H. P. McAdams, E. Samei, J. Dobbins, H. T. Tourassi, and C. E. Ravin, "Recent advances in chest radiography," *Radiology*, vol. 241, no. 3, pp. 663–683, Dec. 2006. doi: 10.1148/radiol.2413051535.

[2] S. Jaeger, A. Karargyris, S. Candemir, L. Folio, J. Siegelman, F. Callaghan, Z. Xue, K. Palaniappan, R. K. Singh, P. K. Antani, G. Thoma, Y. X. Wang, P. X. Lu, and C. J. McDonald, "Automatic screening for tuberculosis in chest radiographs: a survey," *Quantitative Imaging in Medicine and Surgery*, vol. 3, no. 2, pp. 89–99, Apr. 2013. doi: 10.3978/j.issn.2223-4292.2013.04.03.

[3] M. R. K. Mookiah, S. Acharya, C. K. Chua, C. M. Lim, E. Y. K. Ng, and A. Laude, "Computer-aided diagnosis of diabetic retinopathy: A review," *Computers in Biology and Medicine*, vol. 43, no. 12, pp. 2136–2155, Dec. 2013. doi: 10.1016/j.compbiomed.2013.10.007.

[4] S. Candemir, S. Jaeger, K. Palaniappan, J. P. Musco, R. K. Singh, Z. Xue, A. Karargyris, P. K. Antani, G. Thoma, and C. J. McDonald, "Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration," *IEEE Transactions on Medical Imaging*, vol. 33, no. 2, pp. 577–590, Feb. 2014. doi: 10.1109/TMI.2013.2290495.

[5] B. van Ginneken, B. M. ter Haar Romeny, and M. A. Viergever, "Computer-aided diagnosis in chest radiography: a survey," *IEEE Transactions on Medical Imaging*, vol. 20, no. 12, pp. 1228–1241, Dec. 2001. doi: 10.1109/42.974922.

[6] A. A. Novikov, D. Lenis, D. Major, J. Hladůvka, M. Wimmer, and K. Bühler, "Fully convolutional architectures for multiclass segmentation in chest radiographs," *IEEE Transactions on Medical Imaging*, vol. 37, no. 8, pp. 1865–1876, Aug. 2018. doi: 10.1109/TMI.2018.2806086.

[7] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Proc. Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, ser. Lecture Notes in Computer Science, vol. 9351. Cham: Springer, Oct. 2015, pp. 234–241. doi: 10.1007/978-3-319-24574-4_28.

[8] Z. Zhang, Q. Liu, and Y. Wang, "Road extraction by deep residual U-Net," *IEEE Geoscience and Remote Sensing Letters*, vol. 15, no. 5, pp. 749–753, May 2018. doi: 10.1109/LGRS.2018.2802944.

[9] O. Oktay, J. Schlemper, L. Le Folgoc, M. Lee, M. Heinrich, K. Misawa, K. Mori, S. McDonagh, N. Y. Hammerla, B. Kainz, B. Glocker, and D. Rueckert, "Attention U-Net: Learning where to look for the pancreas," in *Proc. Medical Imaging with Deep Learning (MIDL)*, Amsterdam, Netherlands, Jul. 2018. arXiv:1804.03999.

[10] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, "UNet++: A nested U-Net architecture for medical image segmentation," in *Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support (DLMIA/ML-CDS)*, ser. LNCS, vol. 11045. Cham: Springer, Sep. 2018, pp. 3–11. doi: 10.1007/978-3-030-00889-5_1.

[11] G. Litjens, T. Kooi, B. E. Bejnordi, A. A. A. Setio, F. Ciompi, M. Ghafoorian, J. A. W. M. van der Laak, B. van Ginneken, and C. I. Sánchez, "A survey on deep learning in medical image analysis," *Medical Image Analysis*, vol. 42, pp. 60–88, Dec. 2017. doi: 10.1016/j.media.2017.07.005.

[12] T. F. Cootes, C. J. Taylor, D. H. Cooper, and J. Graham, "Active shape models—their training and application," *Computer Vision and Image Understanding*, vol. 61, no. 1, pp. 38–59, Jan. 1995. doi: 10.1006/cviu.1995.1004.

[13] T. F. Cootes, G. J. Edwards, and C. J. Taylor, "Active appearance models," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 23, no. 6, pp. 681–685, Jun. 2001. doi: 10.1109/34.927467.

[14] Y. Y. Boykov and M.-P. Jolly, "Interactive graph cuts for optimal boundary & region segmentation of objects in N-D images," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, Vancouver, BC, Canada, Jul. 2001, pp. 105–112. doi: 10.1109/ICCV.2001.937505.

[15] J. Long, E. Shelhamer, and T. Darrell, "Fully convolutional networks for semantic segmentation," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Boston, MA, USA, Jun. 2015, pp. 3431–3440. doi: 10.1109/CVPR.2015.7298965.

[16] F. Shamrat, R. Ranjan, K. Hasib, A. Yadav, and A. H. Siddique, "A performance comparison of deep learning models for lung segmentation on chest X-ray images," in *Proc. International Conference on Computing, Communication, and Intelligent Systems (ICCCIS)*, Greater Noida, India, Feb. 2021, pp. 675–680. doi: 10.1109/ICCCIS51004.2021.9397194.

[17] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, USA, Jun. 2016, pp. 770–778. doi: 10.1109/CVPR.2016.90.

[18] F. I. Diakogiannis, F. Waldner, P. Caccetta, and C. Wu, "ResUNet-a: A deep learning framework for semantic segmentation of remotely sensed data," *ISPRS Journal of Photogrammetry and Remote Sensing*, vol. 162, pp. 94–114, Apr. 2020. doi: 10.1016/j.isprsjprs.2020.01.013.

[19] S. Jaeger, S. Candemir, S. Antani, Y.-X. J. Wáng, P.-X. Lu, and G. Thoma, "Two public chest X-ray datasets for computer-aided screening of pulmonary diseases," *Quantitative Imaging in Medicine and Surgery*, vol. 4, no. 6, pp. 475–477, Dec. 2014. doi: 10.3978/j.issn.2223-4292.2014.11.20.

[20] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," in *Proc. International Conference on Machine Learning (ICML)*, New York, NY, USA, Jun. 2016, pp. 1050–1059.
