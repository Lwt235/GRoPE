// Copyright (c) 2024-2026 CausalVGGT Authors
// SPDX-License-Identifier: Apache-2.0
//
// MergerWrapper: Tensor-owning wrapper for KV Merger
//
// This class follows the diff-gaussian-rasterization pattern:
// - Wrapper layer owns all torch::Tensor storage
// - Backend layer operates on raw pointers only
// - Pointer views are refreshed after any tensor reallocation

#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <map>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "include/merger_types.h"

namespace causalvggt {

class MergerWrapper {
public:
    MergerWrapper(
        int64_t num_heads,
        int64_t head_dim,
        int64_t pivot_cap,
        int64_t budget_cap,
        int64_t init_voxels,
        torch::Dtype dtype,
        torch::Device device,
        bool seg_mode = false);
    
    ~MergerWrapper();
    
    MergerWrapper(const MergerWrapper&) = delete;
    MergerWrapper& operator=(const MergerWrapper&) = delete;
    MergerWrapper(MergerWrapper&&) = default;
    MergerWrapper& operator=(MergerWrapper&&) = default;
    
    // Core operations - Public API takes torch::Tensor, calls backend with raw ptrs
    void insert_and_merge(
        torch::Tensor K_new,
        torch::Tensor V_new,
        torch::Tensor S_new,
        torch::Tensor VX_new,
        int64_t num_voxels,
        double sim_thresh = 0.75,
        double replace_thresh = 0.5,
        double score_thresh = 0.2);
    
    /**
     * Insert and merge with pre-computed row indices.
     * 
     * Unlike insert_and_merge(), this method:
     * - Takes pre-computed rows [E] (Python handles overflow concat + row computation)
     * - Returns overflow tokens as output (Python manages overflow between calls)
     * - Does NOT touch internal overflow_* state
     * 
     * @param rows Pre-computed row indices [E] int64, where row = head * V_alloc + voxel
     * @param K_new Key vectors [E, D] in pool dtype (fp16/bf16)
     * @param V_new Value vectors [E, D] in pool dtype
     * @param S_new Scores [E] float32
     * @param sim_thresh Similarity threshold for merging (default 0.75)
     * @param replace_thresh Replace threshold (default 0.5)
     * @param score_thresh Score threshold ratio (default 0.2)
     * @return Tuple of (K_over, V_over, S_over, rows_over) - overflow tokens
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    insert_and_merge_with_rows(
        torch::Tensor rows,
        torch::Tensor K_new,
        torch::Tensor V_new,
        torch::Tensor S_new,
        double sim_thresh = 0.75,
        double replace_thresh = 0.5,
        double score_thresh = 0.2);
    
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    retrieve(
        torch::Tensor voxel_ids,
        int64_t retrieve_size = -1,
        int64_t used_voxel_limit = -1);
    
    /**
     * Retrieve buffer (unmerged) tokens from the buffer pool.
     * 
     * Unlike pivot tokens which have log(C) bias, buffer tokens have bias=0.0.
     * Use this in conjunction with retrieve() when return_buf=True is requested.
     * 
     * @param voxel_ids [Q] voxel indices to retrieve
     * @param buf_retrieve_size Output size per head (default Q*B)
     * @param used_voxel_limit Upper bound on valid voxel indices
     * @return (K_buf, V_buf, M_buf, bias_buf) tensors of shape [H, buf_retrieve_size, ...]
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    retrieve_buf(
        torch::Tensor voxel_ids,
        int64_t buf_retrieve_size = -1,
        int64_t used_voxel_limit = -1);
    
    void ensure_capacity(int64_t num_voxels);
    void reset();
    
    /**
     * Validate pool state consistency after operations.
     * 
     * This method checks:
     * - All slots in use have valid row pointers
     * - Row states match pool occupancy
     * - No orphaned data in pools
     * - Free stack contains valid entries without duplicates
     * 
     * @param check_buffer If true, validate buffer pool (default: true)
     * @param check_pivot If true, validate pivot pool (default: true)
     * @return A tuple of (is_valid, error_message)
     *         is_valid is true if all checks pass
     *         error_message contains details about any errors found
     */
    std::tuple<bool, std::string> validate_pool_state(
        bool check_buffer = true,
        bool check_pivot = true) const;
    
    // Accessors
    int64_t num_heads() const { return config_.H; }
    int64_t head_dim() const { return config_.D; }
    int64_t pivot_cap() const { return config_.P; }
    int64_t budget_cap() const { return config_.B; }
    int64_t voxel_alloc() const { return config_.V_alloc; }
    int64_t total_rows() const { return config_.S_tot; }
    
    torch::Tensor buffer_pool_K() const { return buf_pool_K_; }
    torch::Tensor buffer_pool_V() const { return buf_pool_V_; }
    torch::Tensor buffer_pool_S() const { return buf_pool_S_; }
    torch::Tensor buffer_pool_M() const { return buf_pool_M_; }
    
    torch::Tensor pivot_pool_K() const { return piv_pool_K_; }
    torch::Tensor pivot_pool_V() const { return piv_pool_V_; }
    torch::Tensor pivot_pool_W() const { return piv_pool_W_; }
    torch::Tensor pivot_pool_S() const { return piv_pool_S_; }
    torch::Tensor pivot_pool_C() const { return piv_pool_C_; }
    torch::Tensor pivot_pool_M() const { return piv_pool_M_; }
    
    torch::Tensor buffer_row_ptr() const { return buf_row_ptr_; }
    torch::Tensor buffer_row_state() const { return buf_row_state_; }
    torch::Tensor buffer_row_count() const { return buf_row_count_; }
    
    torch::Tensor pivot_row_ptr() const { return piv_row_ptr_; }
    torch::Tensor pivot_row_state() const { return piv_row_state_; }
    
    bool has_overflow() const;
    int64_t overflow_count() const;
    int64_t dropped_overflow_count() const;
    int64_t dropped_overflow_last() const;
    int64_t valid_pivot_count() const;
    int64_t workspace_bytes() const;
    
    std::map<std::string, int64_t> pool_stats() const;

    /** Return and clear diagnostic messages (e.g. [SEG-EXPAND], [SEG-WARN]) for Python to log. */
    std::vector<std::string> take_diagnostics();

    const backend::MergerConfig& config() const { return config_; }
    const backend::MergerViews& views() const { return views_; }

private:
    backend::MergerConfig config_;
    backend::MergerViews views_;
    
    torch::Dtype dtype_;
    torch::Device device_;
    int64_t buf_pool_cap_;
    int64_t piv_pool_cap_;
    int64_t buf_pool_growth_;  // Minimum growth increment for buffer pool expansion
    int64_t piv_pool_growth_;  // Minimum growth increment for pivot pool expansion
    
    // Buffer Pool - torch::Tensor storage (ownership)
    torch::Tensor buf_pool_K_;      // [buf_pool_cap, B, D]
    torch::Tensor buf_pool_V_;      // [buf_pool_cap, B, D]
    torch::Tensor buf_pool_S_;      // [buf_pool_cap, B]
    torch::Tensor buf_pool_M_;      // [buf_pool_cap, B]
    
    torch::Tensor buf_row_ptr_;     // [S_tot]
    torch::Tensor buf_row_state_;   // [S_tot]
    torch::Tensor buf_row_count_;   // [S_tot]
    
    torch::Tensor buf_free_stack_;  // [buf_pool_cap]
    torch::Tensor buf_free_top_;    // [1]
    
    // Pivot Pool - torch::Tensor storage (ownership)
    torch::Tensor piv_pool_K_;      // [piv_pool_cap, P, D]
    torch::Tensor piv_pool_V_;      // [piv_pool_cap, P, D]
    torch::Tensor piv_pool_W_;      // [piv_pool_cap, P]
    torch::Tensor piv_pool_S_;      // [piv_pool_cap, P]
    torch::Tensor piv_pool_C_;      // [piv_pool_cap, P]
    torch::Tensor piv_pool_M_;      // [piv_pool_cap, P]
    torch::Tensor piv_pool_K_seed_; // [piv_pool_cap, P, D]
    torch::Tensor piv_pool_S_seed_; // [piv_pool_cap, P]
    
    torch::Tensor piv_row_ptr_;     // [S_tot]
    torch::Tensor piv_row_state_;   // [S_tot]
    torch::Tensor piv_row_count_;   // [S_tot]
    
    torch::Tensor piv_free_stack_;  // [piv_pool_cap]
    torch::Tensor piv_free_top_;    // [1]
    
    // Overflow Storage
    torch::Tensor overflow_K_;          // [overflow_cap_, D]
    torch::Tensor overflow_V_;          // [overflow_cap_, D]
    torch::Tensor overflow_S_;          // [overflow_cap_]
    torch::Tensor overflow_rows_;       // [overflow_cap_]
    torch::Tensor overflow_count_dev_;  // [1]
    torch::Tensor overflow_dropped_count_dev_;  // [1] int64, cumulative dropped by clamp
    torch::Tensor overflow_dropped_last_dev_;   // [1] int64, dropped in last insert_and_merge
    int64_t overflow_cap_;              // Current overflow buffer capacity
    int64_t overflow_count_host_;       // Actual overflow token count (for workspace sizing)
    
    // Workspace (single contiguous buffer for backend::MergerWorkspace::fromChunk())
    torch::Tensor workspace_;
    int64_t workspace_E_max_;
    bool workspace_skip_packed_;
    bool workspace_skip_combined_kv_;
    int64_t workspace_new_piv_cap_;
    
    // Retrieve workspace (persistent, sized for max Q seen)
    void* retrieve_ws_buffer_;
    size_t retrieve_ws_size_;
    int64_t retrieve_ws_Q_;
    
    // Internal Methods
    void init_pools();
    void init_row_metadata_only();  // Seg-mode: only row_state/row_count, no pool data
    void update_views();  // Key method: refresh raw pointers from tensor data_ptr()
    void expand_row_metadata(int64_t old_alloc, int64_t new_alloc);
    void expand_buf_pool_capacity(int64_t old_cap, int64_t new_cap);
    void expand_piv_pool_capacity(int64_t old_cap, int64_t new_cap);
    void grow_overflow_if_needed(int64_t needed, cudaStream_t stream);
    void ensure_workspace(int64_t E_max,
                          bool skip_packed = false,
                          bool skip_combined_kv = false,
                          int64_t new_piv_cap = -1);
    backend::MergerWorkspace get_workspace(int64_t E_max,
                                           bool skip_packed = false,
                                           bool skip_combined_kv = false,
                                           int64_t new_piv_cap = -1);
    void ensure_retrieve_workspace(int64_t Q, cudaStream_t stream);
    void ensure_pool_slots(cudaStream_t stream);  // Auto-grow pool when free slots are low
    void print_pool_stats(cudaStream_t stream);       // Print contiguous pool diagnostics
    void print_seg_pool_stats(cudaStream_t stream);   // Print segmented pool diagnostics

    // =========================================================================
    // Segmented pool storage (1 token = 1 segment, no internal fragmentation)
    // =========================================================================
    bool use_seg_mode_ = false;
    backend::SegPoolViews seg_views_;

    // Segmented Buffer Pool
    torch::Tensor seg_buf_K_;           // [seg_buf_cap, D]
    torch::Tensor seg_buf_V_;           // [seg_buf_cap, D]
    torch::Tensor seg_buf_S_;           // [seg_buf_cap]
    torch::Tensor seg_buf_free_stack_;  // [seg_buf_cap]
    torch::Tensor seg_buf_free_top_;    // [1]
    torch::Tensor buf_row_seg_;         // [S_tot, B]
    // buf_row_count_, buf_row_state_ reused from contiguous pool

    // Segmented Pivot Pool
    torch::Tensor seg_piv_K_;           // [seg_piv_cap, D]
    torch::Tensor seg_piv_V_;           // [seg_piv_cap, D]
    torch::Tensor seg_piv_W_;           // [seg_piv_cap]
    torch::Tensor seg_piv_S_;           // [seg_piv_cap]
    torch::Tensor seg_piv_C_;           // [seg_piv_cap]
    torch::Tensor seg_piv_K_seed_;      // [seg_piv_cap, D]
    torch::Tensor seg_piv_S_seed_;      // [seg_piv_cap]
    torch::Tensor seg_piv_free_stack_;  // [seg_piv_cap]
    torch::Tensor seg_piv_free_top_;    // [1]
    torch::Tensor piv_row_seg_;         // [S_tot, P]
    // piv_row_count_, piv_row_state_ reused from contiguous pool

    int64_t seg_buf_cap_ = 0;
    int64_t seg_piv_cap_ = 0;
    int64_t seg_buf_growth_ = 0;
    int64_t seg_piv_growth_ = 0;

    // Pivot zone allocator (Morton-based spatial locality)
    torch::Tensor piv_zone_top_;        // [Z] per-zone atomic stack tops
    torch::Tensor voxel_zone_map_;      // [V_alloc] voxel_id -> zone_id

    void init_seg_pools();
    void update_seg_views();
    void expand_seg_buf_pool(int64_t old_cap, int64_t new_cap);
    void expand_seg_piv_pool(int64_t old_cap, int64_t new_cap);
    void expand_seg_row_metadata(int64_t old_S_tot, int64_t new_S_tot);
    void ensure_seg_pool_slots(cudaStream_t stream);

    void push_diagnostic(const char* fmt, ...);

    std::vector<std::string> diagnostic_messages_;

public:
    // Segmented pool public API
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    insert_and_merge_with_rows_seg(
        torch::Tensor rows,
        torch::Tensor K_new,
        torch::Tensor V_new,
        torch::Tensor S_new,
        double sim_thresh = 0.75,
        double replace_thresh = 0.5,
        double score_thresh = 0.2);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    retrieve_seg(
        torch::Tensor voxel_ids,
        int64_t retrieve_size = -1,
        int64_t used_voxel_limit = -1);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    retrieve_buf_seg(
        torch::Tensor voxel_ids,
        int64_t buf_retrieve_size = -1,
        int64_t used_voxel_limit = -1);

    bool is_seg_mode() const { return use_seg_mode_; }
    void set_seg_mode(bool enabled);

    void set_voxel_zones(torch::Tensor zones);

public:
    // round049: observation/serialization only; no policy or kernel changes.
    std::map<std::string, torch::Tensor> audit_tensors() const {
        std::map<std::string, torch::Tensor> out;
        if (buf_pool_K_.defined()) out["buf_pool_K_"] = buf_pool_K_;
        if (buf_pool_V_.defined()) out["buf_pool_V_"] = buf_pool_V_;
        if (buf_pool_S_.defined()) out["buf_pool_S_"] = buf_pool_S_;
        if (buf_pool_M_.defined()) out["buf_pool_M_"] = buf_pool_M_;
        if (buf_row_ptr_.defined()) out["buf_row_ptr_"] = buf_row_ptr_;
        if (buf_row_state_.defined()) out["buf_row_state_"] = buf_row_state_;
        if (buf_row_count_.defined()) out["buf_row_count_"] = buf_row_count_;
        if (buf_free_stack_.defined()) out["buf_free_stack_"] = buf_free_stack_;
        if (buf_free_top_.defined()) out["buf_free_top_"] = buf_free_top_;
        if (piv_pool_K_.defined()) out["piv_pool_K_"] = piv_pool_K_;
        if (piv_pool_V_.defined()) out["piv_pool_V_"] = piv_pool_V_;
        if (piv_pool_W_.defined()) out["piv_pool_W_"] = piv_pool_W_;
        if (piv_pool_S_.defined()) out["piv_pool_S_"] = piv_pool_S_;
        if (piv_pool_C_.defined()) out["piv_pool_C_"] = piv_pool_C_;
        if (piv_pool_M_.defined()) out["piv_pool_M_"] = piv_pool_M_;
        if (piv_pool_K_seed_.defined()) out["piv_pool_K_seed_"] = piv_pool_K_seed_;
        if (piv_pool_S_seed_.defined()) out["piv_pool_S_seed_"] = piv_pool_S_seed_;
        if (piv_row_ptr_.defined()) out["piv_row_ptr_"] = piv_row_ptr_;
        if (piv_row_state_.defined()) out["piv_row_state_"] = piv_row_state_;
        if (piv_row_count_.defined()) out["piv_row_count_"] = piv_row_count_;
        if (piv_free_stack_.defined()) out["piv_free_stack_"] = piv_free_stack_;
        if (piv_free_top_.defined()) out["piv_free_top_"] = piv_free_top_;
        if (overflow_K_.defined()) out["overflow_K_"] = overflow_K_;
        if (overflow_V_.defined()) out["overflow_V_"] = overflow_V_;
        if (overflow_S_.defined()) out["overflow_S_"] = overflow_S_;
        if (overflow_rows_.defined()) out["overflow_rows_"] = overflow_rows_;
        if (overflow_count_dev_.defined()) out["overflow_count_dev_"] = overflow_count_dev_;
        if (overflow_dropped_count_dev_.defined()) out["overflow_dropped_count_dev_"] = overflow_dropped_count_dev_;
        if (overflow_dropped_last_dev_.defined()) out["overflow_dropped_last_dev_"] = overflow_dropped_last_dev_;
        if (workspace_.defined()) out["workspace_"] = workspace_;
        if (seg_buf_K_.defined()) out["seg_buf_K_"] = seg_buf_K_;
        if (seg_buf_V_.defined()) out["seg_buf_V_"] = seg_buf_V_;
        if (seg_buf_S_.defined()) out["seg_buf_S_"] = seg_buf_S_;
        if (seg_buf_free_stack_.defined()) out["seg_buf_free_stack_"] = seg_buf_free_stack_;
        if (seg_buf_free_top_.defined()) out["seg_buf_free_top_"] = seg_buf_free_top_;
        if (buf_row_seg_.defined()) out["buf_row_seg_"] = buf_row_seg_;
        if (seg_piv_K_.defined()) out["seg_piv_K_"] = seg_piv_K_;
        if (seg_piv_V_.defined()) out["seg_piv_V_"] = seg_piv_V_;
        if (seg_piv_W_.defined()) out["seg_piv_W_"] = seg_piv_W_;
        if (seg_piv_S_.defined()) out["seg_piv_S_"] = seg_piv_S_;
        if (seg_piv_C_.defined()) out["seg_piv_C_"] = seg_piv_C_;
        if (seg_piv_K_seed_.defined()) out["seg_piv_K_seed_"] = seg_piv_K_seed_;
        if (seg_piv_S_seed_.defined()) out["seg_piv_S_seed_"] = seg_piv_S_seed_;
        if (seg_piv_free_stack_.defined()) out["seg_piv_free_stack_"] = seg_piv_free_stack_;
        if (seg_piv_free_top_.defined()) out["seg_piv_free_top_"] = seg_piv_free_top_;
        if (piv_row_seg_.defined()) out["piv_row_seg_"] = piv_row_seg_;
        if (piv_zone_top_.defined()) out["piv_zone_top_"] = piv_zone_top_;
        if (voxel_zone_map_.defined()) out["voxel_zone_map_"] = voxel_zone_map_;
        return out;
    }
    int64_t audit_raw_bytes() const { return static_cast<int64_t>(retrieve_ws_size_); }
    std::map<std::string, torch::Tensor> export_state() const {
        auto out = audit_tensors();
        out["__config"] = torch::from_blob((void*)&config_, {(int64_t)sizeof(config_)}, torch::TensorOptions().dtype(torch::kUInt8)).clone();
        out["__scalars"] = torch::tensor(std::vector<int64_t>{(int64_t)buf_pool_cap_,(int64_t)piv_pool_cap_,(int64_t)buf_pool_growth_,(int64_t)piv_pool_growth_,(int64_t)overflow_cap_,(int64_t)overflow_count_host_,(int64_t)workspace_E_max_,(int64_t)workspace_skip_packed_,(int64_t)workspace_skip_combined_kv_,(int64_t)workspace_new_piv_cap_,(int64_t)use_seg_mode_,(int64_t)seg_buf_cap_,(int64_t)seg_piv_cap_,(int64_t)seg_buf_growth_,(int64_t)seg_piv_growth_}, torch::TensorOptions().dtype(torch::kInt64));
        if (retrieve_ws_size_) out["__retrieve"] = torch::from_blob(retrieve_ws_buffer_, {(int64_t)retrieve_ws_size_}, torch::TensorOptions().dtype(torch::kUInt8).device(device_));
        out["__retrieve_Q"] = torch::tensor({retrieve_ws_Q_}, torch::TensorOptions().dtype(torch::kInt64));
        return out;
    }
    void import_state(const std::map<std::string, torch::Tensor>& in) {
        auto cfg = in.at("__config").cpu().contiguous();
        TORCH_CHECK(cfg.numel() == sizeof(config_));
        std::memcpy(&config_, cfg.data_ptr(), sizeof(config_));
        auto values = in.at("__scalars").cpu().contiguous();
        auto s = values.data_ptr<int64_t>();
        buf_pool_cap_ = s[0];
        piv_pool_cap_ = s[1];
        buf_pool_growth_ = s[2];
        piv_pool_growth_ = s[3];
        overflow_cap_ = s[4];
        overflow_count_host_ = s[5];
        workspace_E_max_ = s[6];
        workspace_skip_packed_ = s[7];
        workspace_skip_combined_kv_ = s[8];
        workspace_new_piv_cap_ = s[9];
        use_seg_mode_ = s[10];
        seg_buf_cap_ = s[11];
        seg_piv_cap_ = s[12];
        seg_buf_growth_ = s[13];
        seg_piv_growth_ = s[14];
        buf_pool_K_ = in.count("buf_pool_K_") ? in.at("buf_pool_K_") : torch::Tensor();
        buf_pool_V_ = in.count("buf_pool_V_") ? in.at("buf_pool_V_") : torch::Tensor();
        buf_pool_S_ = in.count("buf_pool_S_") ? in.at("buf_pool_S_") : torch::Tensor();
        buf_pool_M_ = in.count("buf_pool_M_") ? in.at("buf_pool_M_") : torch::Tensor();
        buf_row_ptr_ = in.count("buf_row_ptr_") ? in.at("buf_row_ptr_") : torch::Tensor();
        buf_row_state_ = in.count("buf_row_state_") ? in.at("buf_row_state_") : torch::Tensor();
        buf_row_count_ = in.count("buf_row_count_") ? in.at("buf_row_count_") : torch::Tensor();
        buf_free_stack_ = in.count("buf_free_stack_") ? in.at("buf_free_stack_") : torch::Tensor();
        buf_free_top_ = in.count("buf_free_top_") ? in.at("buf_free_top_") : torch::Tensor();
        piv_pool_K_ = in.count("piv_pool_K_") ? in.at("piv_pool_K_") : torch::Tensor();
        piv_pool_V_ = in.count("piv_pool_V_") ? in.at("piv_pool_V_") : torch::Tensor();
        piv_pool_W_ = in.count("piv_pool_W_") ? in.at("piv_pool_W_") : torch::Tensor();
        piv_pool_S_ = in.count("piv_pool_S_") ? in.at("piv_pool_S_") : torch::Tensor();
        piv_pool_C_ = in.count("piv_pool_C_") ? in.at("piv_pool_C_") : torch::Tensor();
        piv_pool_M_ = in.count("piv_pool_M_") ? in.at("piv_pool_M_") : torch::Tensor();
        piv_pool_K_seed_ = in.count("piv_pool_K_seed_") ? in.at("piv_pool_K_seed_") : torch::Tensor();
        piv_pool_S_seed_ = in.count("piv_pool_S_seed_") ? in.at("piv_pool_S_seed_") : torch::Tensor();
        piv_row_ptr_ = in.count("piv_row_ptr_") ? in.at("piv_row_ptr_") : torch::Tensor();
        piv_row_state_ = in.count("piv_row_state_") ? in.at("piv_row_state_") : torch::Tensor();
        piv_row_count_ = in.count("piv_row_count_") ? in.at("piv_row_count_") : torch::Tensor();
        piv_free_stack_ = in.count("piv_free_stack_") ? in.at("piv_free_stack_") : torch::Tensor();
        piv_free_top_ = in.count("piv_free_top_") ? in.at("piv_free_top_") : torch::Tensor();
        overflow_K_ = in.count("overflow_K_") ? in.at("overflow_K_") : torch::Tensor();
        overflow_V_ = in.count("overflow_V_") ? in.at("overflow_V_") : torch::Tensor();
        overflow_S_ = in.count("overflow_S_") ? in.at("overflow_S_") : torch::Tensor();
        overflow_rows_ = in.count("overflow_rows_") ? in.at("overflow_rows_") : torch::Tensor();
        overflow_count_dev_ = in.count("overflow_count_dev_") ? in.at("overflow_count_dev_") : torch::Tensor();
        overflow_dropped_count_dev_ = in.count("overflow_dropped_count_dev_") ? in.at("overflow_dropped_count_dev_") : torch::Tensor();
        overflow_dropped_last_dev_ = in.count("overflow_dropped_last_dev_") ? in.at("overflow_dropped_last_dev_") : torch::Tensor();
        workspace_ = in.count("workspace_") ? in.at("workspace_") : torch::Tensor();
        seg_buf_K_ = in.count("seg_buf_K_") ? in.at("seg_buf_K_") : torch::Tensor();
        seg_buf_V_ = in.count("seg_buf_V_") ? in.at("seg_buf_V_") : torch::Tensor();
        seg_buf_S_ = in.count("seg_buf_S_") ? in.at("seg_buf_S_") : torch::Tensor();
        seg_buf_free_stack_ = in.count("seg_buf_free_stack_") ? in.at("seg_buf_free_stack_") : torch::Tensor();
        seg_buf_free_top_ = in.count("seg_buf_free_top_") ? in.at("seg_buf_free_top_") : torch::Tensor();
        buf_row_seg_ = in.count("buf_row_seg_") ? in.at("buf_row_seg_") : torch::Tensor();
        seg_piv_K_ = in.count("seg_piv_K_") ? in.at("seg_piv_K_") : torch::Tensor();
        seg_piv_V_ = in.count("seg_piv_V_") ? in.at("seg_piv_V_") : torch::Tensor();
        seg_piv_W_ = in.count("seg_piv_W_") ? in.at("seg_piv_W_") : torch::Tensor();
        seg_piv_S_ = in.count("seg_piv_S_") ? in.at("seg_piv_S_") : torch::Tensor();
        seg_piv_C_ = in.count("seg_piv_C_") ? in.at("seg_piv_C_") : torch::Tensor();
        seg_piv_K_seed_ = in.count("seg_piv_K_seed_") ? in.at("seg_piv_K_seed_") : torch::Tensor();
        seg_piv_S_seed_ = in.count("seg_piv_S_seed_") ? in.at("seg_piv_S_seed_") : torch::Tensor();
        seg_piv_free_stack_ = in.count("seg_piv_free_stack_") ? in.at("seg_piv_free_stack_") : torch::Tensor();
        seg_piv_free_top_ = in.count("seg_piv_free_top_") ? in.at("seg_piv_free_top_") : torch::Tensor();
        piv_row_seg_ = in.count("piv_row_seg_") ? in.at("piv_row_seg_") : torch::Tensor();
        piv_zone_top_ = in.count("piv_zone_top_") ? in.at("piv_zone_top_") : torch::Tensor();
        voxel_zone_map_ = in.count("voxel_zone_map_") ? in.at("voxel_zone_map_") : torch::Tensor();
        if (in.count("__retrieve")) {
            auto q = in.at("__retrieve_Q").cpu().item<int64_t>();
            ensure_retrieve_workspace(q, c10::cuda::getCurrentCUDAStream().stream());
            auto src = in.at("__retrieve").contiguous();
            TORCH_CHECK(src.numel() == retrieve_ws_size_);
            cudaMemcpyAsync(retrieve_ws_buffer_, src.data_ptr(), retrieve_ws_size_, cudaMemcpyDeviceToDevice, c10::cuda::getCurrentCUDAStream().stream());
        }
        if (use_seg_mode_) update_seg_views(); else update_views();
    }
};

inline bool has_merger_wrapper() {
    return true;
}

}  // namespace causalvggt
