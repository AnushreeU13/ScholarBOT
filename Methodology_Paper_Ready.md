
# Technical Methodology: Tiered Clinical RAG Architecture

## 1. High-Dimensional Knowledge Representation
The core of the system relies on a dense vector search architecture powered by **MiniLM-L6-v2**, a highly efficient 384-dimensional embedding model optimized for sentence-level semantic similarity [1]. This model ensures compatibility with standard FAISS indices and minimizes inference latency without significant accuracy loss in general domain retrieval tasks.

The Knowledge Base (KB) is partitioned into three isolated indices to prevent semantic pollution across domains:
1.  **Clinical Guidelines Index ($I_{guide}$)**: Segmented into `400`-token chunks with a `50`-token overlap, retaining hierarchical metadata through Structure-Aware Parsing.
2.  **Pharmacological Index ($I_{drug}$)**: FDA SPL labels chunked by section headers (e.g., "Boxed Warnings"), enabling targeted retrieval of safety data.
3.  **Dynamic User Index ($I_{user}$)**: An ephemeral index for real-time RAG operations on user-uploaded documents.

## 2. Structurally Constrained Retrieval Architecture
We implement a **Tiered Retrieval-Augmented Generation** pipeline designed for "Fail-Closed" safety. The retrieval process $R(q)$ is governed by a cascade of task-aware logic gates and metric constraints.

### 2.1. Semantic Routing & Section Boosting
Incoming queries $q$ are classified into intents $C \in \{Guideline, Drug, Mixed\}$ by a semantic router. To mitigate "semantic drift"—where high vector similarity occurs in irrelevant sections (e.g., "Description" instead of "Adverse Reactions")—we apply a **Section Boosting** algorithm:

$$ S'(d, q) = S(d, q) + \alpha \cdot \mathbb{I}(meta(d) \in P_q) $$

Where $S(d, q)$ is the raw cosine similarity and $\alpha = 0.12$ is the boosting hyperparameter. $\mathbb{I}$ is an indicator function that activates if the document chunk $d$ belongs to a preferred section group $P_q$ (e.g., *Adverse Reactions*) inferred from the query token space.

### 2.2. Retrieval Hyperparameters
The system employs a high-recall configuration with $k=8$ (Top-K) to maximize evidence capture. A crucial safety mechanism is the **Abstention Threshold** ($\tau_{abstain}$). If the maximum similarity score of the retrieved set falls below $\tau_{abstain} = 0.35$, the system preemptively aborts generation and returns an `ABSTAIN` token. This threshold is tuned conservatively to favor recall in diverse clinical scenarios.

## 3. Dual-Perspective Gated Generation
The generation module $\Phi(q, D)$ utilizes a **Llama 3 (8B)** instruction-tuned backbone to produce two distinct outputs, safeguarded by verification logic.

### 3.1. Lexical Entailment Verification (Clinician)
The clinical output is subjected to a **Lexical Entailment Gate** that measures the token-level support of every generated claim against the retrieved graph. A claim $c$ is accepted only if its overlap ratio $O(c, D)$ exceeds the entailment threshold $\tau_{entail} = 0.15$.
$$ O(c, D) = \frac{|Tokens(c) \cap Tokens(D)|}{|Tokens(c)|} $$
Statements failing this check are pruned to minimize extrinsic hallucinations.

### 3.2. Semantic Consistency & Safety (Patient)
The patient-facing simplified output is verified against the technical summary using two gates:
1.  **Medical Entity Safety Guard**: A deterministic filter that rejects the patient summary if it introduces *new* Named Entities (drugs, tests, conditions) not present in the clinician summary ($E_{patient} \not\subseteq E_{clinician}$).
2.  **Semantic Consistency Check**: We compute the cosine similarity between the embeddings of the clinical and patient outputs. A score below $\tau_{consist} = 0.72$ triggers a fallback mechanism, ensuring the simplification does not semantically diverge from the technical source.

**References:**
[1] Wang, N., et al. (2021). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *Microsoft Research*.

