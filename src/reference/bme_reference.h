#ifndef BME_REFERENCE_H
#define BME_REFERENCE_H

/**
 * @file bme_reference.h
 * @brief Public C99 API for Phylo2Vec Balanced Minimum Evolution (BME) coordinate
 *        evaluation, O(N) ordered tree encoding/relabelling, Pardi subtree
 *        condensation, and reference BME/FastME tree optimization.
 *
 * Mathematical & algorithmic references refer to:
 *   "Phylo2Vec-Minibatch: phylogenetic inference with uncertainty for millions of sequences"
 *   - Supplementary Section S1 (Proposition S1): Exact O(N^2) coordinate-difference table Delta_{i,j}
 *   - Supplementary Section S2 (Equation S1): Pardi fixed-component BME condensation
 *   - Supplementary Section S4: Consistency of condensed search & two-gauge SPR coverage
 *   - Supplementary Section S5: Implementation details (O(N) ordered decoding & relabelling)
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
#define P2V_RESTRICT __restrict__
extern "C" {
#else
#define P2V_RESTRICT restrict
#endif

/* ============================================================================
 * 1. Exact O(N^2) Phylo2Vec Coordinate-Difference Evaluation (Section S1)
 * ============================================================================ */

/**
 * @brief Computes all (N-1)^2 exact BME coordinate differences Delta_{i,j} in O(N^2) time.
 *
 * For an ordered ("cherry-first") Phylo2Vec vector v of length N-1 satisfying
 * 0 <= v[i] <= i, evaluates the exact change in the unrooted Pauplin BME objective:
 *
 *   Delta_{i,j} = L(T(v^(i <- j)); D) - L(T(v); D),   for 0 <= i <= N-2, 0 <= j <= 2i,
 *
 * including both pendant-edge regrafts (0 <= j <= i) and internal-edge regrafts
 * (i < j <= 2i). Uses the Desper-Gascuel (2002) balanced-average recurrence followed
 * by the Hordijk-Gascuel (2005) SPR shift recurrence along each row i (Proposition S1).
 *
 * @param[in]  v               Ordered Phylo2Vec vector of length N-1 (0 <= v[i] <= i).
 * @param[in]  D               Row-major N x N symmetric distance matrix (permuted to match v).
 * @param[in]  n_leaves        Number of leaves N (N >= 4).
 * @param[out] delta_full_out  Row-major output array of shape (N-1) x (2N-3). Entry
 *                             delta_full_out[i * (2*n_leaves - 3) + j] stores Delta_{i,j}
 *                             for 0 <= j <= 2i, and 0.0 for j > 2i.
 *
 * @note Complexity: O(N^2) arithmetic operations and O(N^2) thread-local workspace.
 */
void calculate_discrete_gradient_full_c(
    const int64_t *P2V_RESTRICT v,
    const double  *P2V_RESTRICT D,
    int32_t                     n_leaves,
    double        *P2V_RESTRICT delta_full_out
);

/**
 * @brief Computes the lower-triangular pendant-edge subset (0 <= j < i) of Delta_{i,j}.
 *
 * @param[in]  v          Ordered Phylo2Vec vector of length N-1 (0 <= v[i] <= i).
 * @param[in]  D          Row-major N x N symmetric distance matrix.
 * @param[in]  n_leaves   Number of leaves N (N >= 4).
 * @param[out] delta_out  Row-major output array of shape (N-1) x (N-2), where entry
 *                        delta_out[i * (n_leaves - 2) + j] stores Delta_{i,j} for 0 <= j < i.
 *
 * @note Complexity: O(N^2) time and O(N^2) space.
 */
void calculate_discrete_gradient_c(
    const int64_t *P2V_RESTRICT v,
    const double  *P2V_RESTRICT D,
    int32_t                     n_leaves,
    double        *P2V_RESTRICT delta_out
);

/* ============================================================================
 * 2. O(N) Tree Bijections & Ordered ("Cherry-First") Relabelling (Section S5)
 * ============================================================================ */

/**
 * @brief Relabels any general Phylo2Vec vector (0 <= v_in[i] <= 2i) into an ordered
 *        Phylo2Vec vector (0 <= v_out[i] <= i) and leaf permutation in O(N) time.
 *
 * Every binary tree admits an ordered Phylo2Vec encoding under a suitable permutation
 * of leaf indices (Penn et al., 2023). In the standard orientation (flip_mode = 0),
 * at every internal vertex the child whose subtree contains the smallest original leaf
 * label retains the parent's label, and the other child receives the next unused label.
 * In the reversed orientation (flip_mode = 1), the roles of the two children are swapped.
 *
 * @param[in]  v_in      General Phylo2Vec vector of length N-1 (0 <= v_in[i] <= 2i).
 * @param[in]  n_leaves  Number of leaves N.
 * @param[in]  flip_mode 0 for standard orientation (gauge 0); 1 for reversed orientation (gauge 1).
 * @param[out] v_out     Output ordered Phylo2Vec vector of length N-1 (0 <= v_out[i] <= i).
 * @param[out] perm_out  Output leaf permutation of length N: computational leaf a in {0..N-1}
 *                       corresponds to original leaf perm_out[a] in {0..N-1}.
 *
 * @note Complexity: O(N) time and O(N) space.
 */
void relabel_cherry_first_c(
    const int64_t *P2V_RESTRICT v_in,
    int32_t                     n_leaves,
    int32_t                     flip_mode,
    int64_t       *P2V_RESTRICT v_out,
    int32_t       *P2V_RESTRICT perm_out
);

/**
 * @brief Encodes a rooted binary tree (given as child arrays) into its unique general
 *        Phylo2Vec vector v_out (0 <= v_out[i] <= 2i) in O(N) time.
 *
 * @param[in]  child1_r  Left child array of length 2N-1 (leaves are 0..N-1, internal nodes
 *                       are N..2N-2, and the root is 2N-2).
 * @param[in]  child2_r  Right child array of length 2N-1.
 * @param[in]  n_leaves  Number of leaves N.
 * @param[out] v_out     Output Phylo2Vec vector of length N-1 (0 <= v_out[i] <= 2i).
 *
 * @note Complexity: O(N) time and O(N) space.
 */
void encode_rooted_to_v_c(
    const int32_t *P2V_RESTRICT child1_r,
    const int32_t *P2V_RESTRICT child2_r,
    int32_t                     n_leaves,
    int64_t       *P2V_RESTRICT v_out
);

/**
 * @brief Encodes an unrooted binary tree (given as a degree-3 adjacency table) into
 *        a general Phylo2Vec vector v_out (0 <= v_out[i] <= 2i) in O(N) time.
 *
 * @param[in]  adj_u     Unrooted adjacency array of length 3 * (2N-2). Each internal node
 *                       u in {N..2N-3} has 3 neighbors at adj_u[3*u + 0..2]; each leaf
 *                       x in {0..N-1} has its pendant neighbor at adj_u[3*x + 0] and -1
 *                       at adj_u[3*x + 1..2].
 * @param[in]  n_leaves  Number of leaves N.
 * @param[out] v_out     Output Phylo2Vec vector of length N-1 (0 <= v_out[i] <= 2i).
 *
 * @note Complexity: O(N) time and O(N) space.
 */
void encode_unrooted_to_v_c(
    const int32_t *P2V_RESTRICT adj_u,
    int32_t                     n_leaves,
    int64_t       *P2V_RESTRICT v_out
);

/* ============================================================================
 * 3. Exact BME Subtree Condensation & Expansion (Section S2, Equation S1)
 * ============================================================================ */

/**
 * @brief Condenses marked subtrees of an N-leaf tree into a K-component skeleton G
 *        and computes the exact K x K condensed BME distance matrix D_eff_out.
 *
 * Implements Pardi (2009, Proposition 6.4.1): for any partition of the leaves into
 * disjoint rooted components S_1, ..., S_K connected by a K-leaf skeleton G,
 *
 *   L(T_G; D) = C + L(G; D_tilde),
 *
 * where C is independent of G and D_tilde[a,b] = sum_{x in S_a} sum_{y in S_b} omega_a(x) omega_b(y) D[x,y]
 * with balanced weights omega_a(x) = 2^{-h_a(x)}.
 *
 * @param[in]  v                   Full N-leaf Phylo2Vec vector of length N-1.
 * @param[in]  D                   Full N x N symmetric distance matrix.
 * @param[in]  n_leaves            Number of leaves N.
 * @param[in]  is_condensed_node   Boolean mask of length 2N-1; non-zero at node u collapses
 *                                 the rooted subtree at u into a single skeleton component.
 * @param[out] n_eff_out           Output number of components K in the condensed skeleton.
 * @param[out] v_eff_out           Output Phylo2Vec vector of the K-leaf skeleton G (length K-1).
 * @param[out] super_node_map_out  Array of length K mapping skeleton leaf a in {0..K-1} to its
 *                                 root node ID in {0..2N-2} of the original tree.
 * @param[out] leaf_to_super_out   Array of length N mapping original leaf x to its component a in {0..K-1}.
 * @param[out] omega_leaf_out      Array of length N storing exact dyadic weight omega_a(x) = 2^{-h_a(x)}.
 * @param[out] D_eff_out           Output K x K condensed distance matrix D_tilde (row-major).
 */
void condense_tree_bme_c(
    const int64_t *P2V_RESTRICT v,
    const double  *P2V_RESTRICT D,
    int32_t                     n_leaves,
    const uint8_t *P2V_RESTRICT is_condensed_node,
    int32_t       *P2V_RESTRICT n_eff_out,
    int64_t       *P2V_RESTRICT v_eff_out,
    int32_t       *P2V_RESTRICT super_node_map_out,
    int32_t       *P2V_RESTRICT leaf_to_super_out,
    double        *P2V_RESTRICT omega_leaf_out,
    double        *P2V_RESTRICT D_eff_out
);

/**
 * @brief Expands an updated K-leaf skeleton v_eff back into a full N-leaf Phylo2Vec vector.
 *
 * @param[in]  v_orig          Original N-leaf Phylo2Vec vector prior to condensation.
 * @param[in]  n_leaves        Total number of leaves N.
 * @param[in]  v_eff           Updated K-leaf skeleton Phylo2Vec vector (length K-1).
 * @param[in]  n_eff           Number of skeleton components K.
 * @param[in]  super_node_map  Component-to-original-node map of length K from condense_tree_bme_c.
 * @param[out] v_expanded_out  Output N-leaf Phylo2Vec vector of length N-1.
 *
 * @note Complexity: O(N) time and O(N) space.
 */
void expand_condensed_tree_to_v_c(
    const int64_t *P2V_RESTRICT v_orig,
    int32_t                     n_leaves,
    const int64_t *P2V_RESTRICT v_eff,
    int32_t                     n_eff,
    const int32_t *P2V_RESTRICT super_node_map,
    int64_t       *P2V_RESTRICT v_expanded_out
);

/**
 * @brief Selects a K-component clade partition on an N-leaf tree for CPU minibatch optimization.
 *
 * @param[in]  v                      Current N-leaf Phylo2Vec vector.
 * @param[in]  n_leaves               Number of leaves N.
 * @param[in]  target_n_eff           Target skeleton component count K.
 * @param[in]  batch_idx              Pass/window index controlling partition granularity.
 * @param[out] is_condensed_node_out  Output mask of length 2N-1 for condense_tree_bme_c.
 */
void select_minibatch_clades_c(
    const int64_t *P2V_RESTRICT v,
    int32_t                     n_leaves,
    int32_t                     target_n_eff,
    int32_t                     batch_idx,
    uint8_t       *P2V_RESTRICT is_condensed_node_out
);

/* ============================================================================
 * 4. End-to-End CPU Reference Tree Optimizers & Workspace Control
 * ============================================================================ */

/**
 * @brief Optimizes an N-leaf tree under the BME criterion in Phylo2Vec space (CPU reference).
 *
 * Alternates O(N^2) coordinate-difference sweeps, path-disjoint batched moves,
 * two-gauge ordered relabelling (standard and reversed), and an SPR completion scan
 * until no improving SPR move remains (Section S4).
 *
 * @param[in]  v_init            Initial Phylo2Vec vector of length N-1.
 * @param[in]  D                 Row-major N x N symmetric distance matrix.
 * @param[in]  n_leaves          Number of leaves N.
 * @param[out] v_out             Output optimized Phylo2Vec vector of length N-1.
 * @param[out] n_grad_evals_out  Optional output pointer receiving number of O(N^2) sweeps.
 * @param[out] elapsed_ms_out    Optional output pointer receiving elapsed wall time in ms.
 * @return                       Final exact Pauplin BME tree length L(T(v_out); D).
 */
double optimize_phylo2vec_bme_c(
    const int64_t *P2V_RESTRICT v_init,
    const double  *P2V_RESTRICT D,
    int32_t                     n_leaves,
    int64_t       *P2V_RESTRICT v_out,
    int32_t       *P2V_RESTRICT n_grad_evals_out,
    double        *P2V_RESTRICT elapsed_ms_out
);

/**
 * @brief CPU reference multi-window condensed minibatch BME optimizer.
 *
 * @param[in]  v_init        Initial Phylo2Vec vector of length N-1.
 * @param[in]  D             Full N x N distance matrix.
 * @param[in]  n_leaves      Number of leaves N.
 * @param[in]  target_n_eff  Target skeleton size K per window.
 * @param[in]  n_batches     Number of condensed windows to run.
 * @param[out] v_out         Output N-leaf Phylo2Vec vector of length N-1.
 * @return                   Final exact Pauplin BME tree length.
 */
double optimize_phylo2vec_bme_minibatch_c(
    const int64_t *P2V_RESTRICT v_init,
    const double  *P2V_RESTRICT D,
    int32_t                     n_leaves,
    int32_t                     target_n_eff,
    int32_t                     n_batches,
    int64_t       *P2V_RESTRICT v_out
);

/**
 * @brief Frees thread-local scratch buffers allocated by the C99 evaluator on the calling thread.
 */
void free_phylo2vec_workspace_c(void);

#ifdef __cplusplus
}
#endif

#endif /* BME_REFERENCE_H */
