/*
 * kcGMLMPop_computeBlock.hpp
 * Computations for a GMLMPop+derivatives (on one GPU).
 *
 * Package GMLM_dmc for dimensionality reduction of neural data.
 *   Population GMLM and indidivual-cell-recordings GMLM are in separate CUDA files.
 *   I decided to do this in order to make the different optimization requirements a little more clean.
 *   
 *  References
 *   Kenneth Latimer & David Freeedman (2021). Low-dimensional encoding of 
 *   decisions in parietal cortex reflects long-term training history.
 *   bioRxiv
 *
 *  Copyright (c) 2021 Kenneth Latimer
 *
 *   This software is distributed under the GNU General Public
 *   License (version 3 or later); please refer to the file
 *   License.txt, included with the software, for details.
 */
#ifndef GMLM_GMLMPop_COMPUTEBLOCK_H
#define GMLM_GMLMPop_COMPUTEBLOCK_H

// #include "kcGMLM_dataStructures.hpp"

namespace kCUDA {

/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
* CLASS GPUGMLMPop_computeBlock <FPTYPE = double, float>
*        This is for data on a single GPU. Holds all the trials and compute space.
*/
template <class FPTYPE> 
class GPUGMLMPop_computeBlock : public GPUGMLM_computeBlock_base<FPTYPE> {
    private:
        // parameter space on GPU (plus page-locked memory copy)
        GPUGMLM_parameters_GPU<FPTYPE> * params;
        
        // dataset
        GPUGMLMPop_dataset_GPU<FPTYPE> * dataset;
        
        // results
        GPUGMLM_results_GPU<FPTYPE> * results;
        
        // streams and handles
        cudaStream_t stream; // main stream for block
        std::vector<cudaStream_t> stream_Groups;
        cublasHandle_t cublasHandle; // main handle for block
        std::vector<cublasHandle_t> cublasHandle_Groups;
        std::vector<cusparseHandle_t> cusparseHandle_Groups;

        size_t cublasWorkspace_size;
        std::vector<size_t> cublasWorkspaces_size;
        FPTYPE * cublasWorkspace;
        std::vector<FPTYPE *> cublasWorkspaces;
        
        size_t dim_J;
        
        bool results_set = false;
        
        cudaEvent_t LL_event;
    public:
        //Primary constructor takes in the GMLMPop setup (as the full GPUGMLMPop class does), but only the specific block data
        GPUGMLMPop_computeBlock(const GPUGMLM_structure_args<FPTYPE> * GMLMPopstructure, const GPUGMLM_GPU_block_args <FPTYPE> * block, const size_t max_trials_, std::shared_ptr<GPUGL_msg> msg_);
        
        //destructor
        ~GPUGMLMPop_computeBlock();
        
        //sync all streams on the block
        inline void syncStreams() {
            this->switchToDevice();
            this->checkCudaErrors( cudaStreamSynchronize(stream), "GPUGMLMPop_computeBlock errors: could not synchronize streams!");
            for(auto jj : stream_Groups) {
                this->checkCudaErrors( cudaStreamSynchronize(jj), "GPUGMLMPop_computeBlock errors: could not synchronize group streams!");
            }
        }
        
        bool loadParams(const GPUGMLM_params<FPTYPE> * params_host, const GPUGMLM_computeOptions<FPTYPE> * opts = NULL);
        
        void computeRateParts(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun);

        void computeLogLike(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun);

        void computeDerivatives(const GPUGMLM_computeOptions<FPTYPE> * opts, const bool isSparseRun);
        
        inline void gatherResults(const GPUGMLM_computeOptions<FPTYPE> * opts) {
            if(results_set) {
                results->gatherResults(params, opts, stream, stream_Groups);
            }
        }
        inline bool addResultsToHost(GPUGMLM_results<FPTYPE>* results_dest, const GPUGMLM_computeOptions<FPTYPE> * opts, const bool reset = false) {
            if(results_set) {
                results->addToHost(params, results_dest, opts, &(dataset->isInDataset_trial), NULL, reset);
            }
            
            return results_set;
        }
};

  
/*The main datastructures for the GMLMPop on the GPU
 *      parameters
 *      results
 *      dataset (all data + pre-allocated compute space)
 *    
 *      I make overly generous use of friend classes to streamline some computation code.
 */
    

/**************************************************************************************************************************************************************
***************************************************************************************************************************************************************
 * DATASET on GPU 
 * 
 */
template <class FPTYPE>
class GPUGMLMPop_dataset_GPU : GPUGL_base {
    private:
        friend class GPUGMLMPop_computeBlock<FPTYPE>; 
        friend class GPUGMLMPop_dataset_Group_GPU<FPTYPE>;
        friend class GPUGMLM_parameters_GPU<FPTYPE>; 
        friend class GPUGMLM_results_GPU<FPTYPE>; 
        friend class GPUGMLM_results_Group_GPU<FPTYPE>; 

        FPTYPE log_dt; // log of bin size
        FPTYPE dt; 
        
        size_t max_trial_length; // max trial length for pre-allocation
        
        size_t max_trials_for_sparse_run;
        
        std::vector<bool> isInDataset_trial;  // size max_trials - if each trial is in this block
        
        GPUData<size_t> * dim_N = NULL;  //X trial lengths (size dim_M x 1)
        
        size_t dim_N_temp; // number of bins in this block for current sparse run (running only some trials)
        
        //spike counts (all trials)
        GPUData<FPTYPE> * Y = NULL; // size   dim_N_total x dim_P
        
        GPUData<FPTYPE> * normalizingConstants_trial = NULL; //X normalizing constant for each trial (dim_M x dim_P)
        
        //linear terms (split by neuron)
        GPUData<FPTYPE> * X_lin = NULL;  // X_lin is dim_N_total x dim_B x dim_P (third dim can be 1 so that all neurons have same linear term)
        
        //indices
        GPUData<unsigned int> * ridx_t_all = NULL;    //X idx of each trial into complete data arrays (like Y)
        
        GPUData<unsigned int> * ridx_st_sall = NULL;       // idx of each trial (indexed by current run on non-zero weighted trials) into sparse index
        GPUData<unsigned int> * ridx_sa_all = NULL;        // idx of sparse run indices into complete data arrays
        GPUData<unsigned int> * ridx_a_all = NULL;        // dummy array for more transparent full/sparse runs
        GPUData<unsigned int> * ridx_a_all_c = NULL;
        GPUData<unsigned int> * ridx_t_all_c = NULL;
        
        GPUData<unsigned int> * id_t_trial = NULL;  //X total trial number of each trial (total is across all GPUs, indexes into trial_weights and trialLL)   size is dim_M x 1
        
        GPUData<unsigned int> * id_a_trialM;  //X local trial index of each observation  size is dim_N_total x 1
                            // the M says that these numbers are 0:(dim_M-1). This is a reminder that it's different numbers than id_t_trial (without the M)
        
        //compute space
        GPUData<FPTYPE> * lambda = NULL; // dim_N_total x dim_P x dim_J, contribution to the log rate for each group
        
        GPUData<FPTYPE> *  LL = NULL; // size   dim_N_total x dim_P
        GPUData<FPTYPE> * dLL = NULL; // size   dim_N_total x dim_P
        
        GPUData<FPTYPE> *  dW_trial = NULL; // size   dim_M x dim_P
        
        GPUData<FPTYPE> * X_lin_temp = NULL; // temporary X_lin space for sparse computations, size  max_trial_length*max_trials_for_sparse_run x dim_B x dim_P
        
        //groups
        std::vector<GPUGMLMPop_dataset_Group_GPU<FPTYPE> *> Groups;
        
    public:
        
        //Primary constructor takes in all the block data and GMLMPop setup
        GPUGMLMPop_dataset_GPU(const GPUGMLM_structure_args<FPTYPE> * GMLMPopstructure, const GPUGMLM_GPU_block_args <FPTYPE> * block, const size_t max_trials_,  const cudaStream_t stream, const std::vector<cusparseHandle_t> & cusparseHandle_Groups, std::shared_ptr<GPUGL_msg> msg_);
            
        //destructor
        ~GPUGMLMPop_dataset_GPU();
        
        inline bool isSparseRun(const GPUGMLM_parameters_GPU<FPTYPE> * params) {
            return params->getNumberOfNonzeroWeights() <= max_trials_for_sparse_run;
        }


        inline cudaError_t waitForGroups_LL(cudaStream_t stream) {
            cudaError_t ce = cudaSuccess;
            for(auto jj : Groups) {
                ce = cudaStreamWaitEvent(stream, jj->group_LL_event, 0);
                if(ce != cudaSuccess) {
                    break;
                }
            }
            return ce;
        }
        
        //dimensions
        inline size_t dim_J() const { //number of coefficient tensor groups
            return Groups.size();
        }
        inline size_t dim_P() const { // number of neurons
            return Y->getSize(1);
        }
        inline size_t dim_B() const { // number of linear coefs
            return X_lin->getSize(1);
        }
        inline size_t dim_M() const { // number of trials
            return dim_N->getSize(0);
        }
        inline size_t dim_N_total() const { // number of bins in this block (sum dim_N)
            return Y->getSize(0);
        }
        inline size_t max_trials() const {
            return isInDataset_trial.size();
        }
};

//The tensor groups contain both the data and the compute operations
//   GPUGMLMPop_dataset_Group_GPU is an abstract class
template <class FPTYPE>
class GPUGMLMPop_dataset_Group_GPU : protected GPUGL_base {
    protected:
        friend class GPUGMLMPop_computeBlock<FPTYPE>; 
        friend class GPUGMLMPop_dataset_GPU<FPTYPE>; 
        
        const GPUGMLMPop_dataset_GPU<FPTYPE> * parent;
        
        const int groupNum;
        
        size_t dim_A;          // number of events for this group
        
        //regressors
        std::vector<GPUData<FPTYPE> *> X; // the regressors for each facotr group (dim_D x 1)
                    // if isShared[dd]==false -  X[dd] is  parent->dim_N_total x dim_F[dd] x (dim_A or 1), if third dim is 1 and dim_A > 0, uses same regressors for each event
                    // if isShared[dd]==true  -  X[dd] is  dim_X[ss] x dim_F[dd]
        
        std::vector<GPUData<int> *> iX; //indices into the rows of X[dd] for any isShared[dd]==true (dim_D x 1)
                    // if isShared[dd]==false -  iX_shared[dd] is  empty
                    // if isShared[dd]==true  -  iX_shared[dd] is  parent->dim_N_total x dim_A
        
        GPUData<bool> * isShared = NULL; // if the regressors for each factor group are shared (global) or local (dim_D x 1)
        GPUData<bool> * isSharedIdentity = NULL; // if the shared regressors for each factor group are the identity matrix (dim_D x 1)
        
        
        //compute space
        //   regressors times parameters
        std::vector<GPUData<FPTYPE> *>  XF; // each X times T (dim_D of them)
                    // if isShared[dd]==false -  XF[dd] is  parent->dim_N_total x dim_R_max x (dim_A or 1)
                    // if isShared[dd]==true  -  XF[dd] is  dim_X[dd] x dim_R_max

        
        //  compute space for derivatives
        GPUData<FPTYPE> * lambda_v = NULL; // dim_N_total x dim_R_max
        
        std::vector<GPUData<FPTYPE> *> lambda_d; // (dim_N_total x dim_R_max) x (dim_A or 1)
        
        GPUData<FPTYPE> * phi_d = NULL; // max({dim_N_total, X[dd].getSize(0) : isShared[dd]}) x dim_R_max
        
        //for sparse runs
        std::vector<GPUData<FPTYPE> *> X_temp;
        
        // sparse matrices for shared regressor derivatives
        std::vector<GPUData<int> *> spi_rows;
        std::vector<GPUData<int> *> spi_cols;
        
        std::vector<GPUData<FPTYPE> *>  spi_data;
        
        std::vector<cusparseSpMatDescr_t*> spi_S;
        std::vector<cusparseDnVecDescr_t*> spi_phi_d;
        std::vector<cusparseDnVecDescr_t*> spi_lambda_d;
        std::vector<GPUData<char> *>  spi_buffer;
        std::vector<size_t> spi_buffer_size;
        
        cudaEvent_t group_LL_event;
    public:
        //constructor
        GPUGMLMPop_dataset_Group_GPU(const int groupNum_, const GPUGMLM_structure_Group_args<FPTYPE> * GMLMPopGroupStructure, const std::vector<GPUGMLM_trial_args <FPTYPE> *> trials, const GPUGMLMPop_dataset_GPU<FPTYPE> * parent_, const cudaStream_t stream, const cusparseHandle_t & cusparseHandle);
        
        //destructor
        ~GPUGMLMPop_dataset_Group_GPU();
        
        
        void multiplyCoefficients(const bool isSparseRun, const bool update_weights, const GPUGMLM_parameters_Group_GPU<FPTYPE> * params, const cudaStream_t stream, const cublasHandle_t cublasHandle, cudaEvent_t & paramsLoaded);
        void getGroupRate(const bool isSparseRun, const GPUGMLM_parameters_Group_GPU<FPTYPE> * params, const GPUGMLM_group_computeOptions * opts, const cudaStream_t stream, const cublasHandle_t cublasHandle);
        void computeDerivatives(GPUGMLM_results_Group_GPU<FPTYPE> * results, const bool isSparseRun,const bool update_weights, GPUGMLM_parameters_Group_GPU<FPTYPE> * params, const GPUGMLM_group_computeOptions * opts, const cudaStream_t stream, const cublasHandle_t cublasHandle, const cusparseHandle_t cusparseHandle, cudaEvent_t & main_LL_event);

        //dimensions
        inline size_t dim_P() const  {
            return parent->dim_P();
        }
        inline size_t dim_X(int ff) const  {
            return X[ff]->getSize(0);
        }
        inline size_t dim_F(int idx) const  {
            return X[idx]->getSize(1);
        }
        inline size_t dim_D() const { // number of factors for tensor decomposition (if == dim_S, than is full CP, otherwise parts are full tensors
            return X.size();
        }
        inline size_t dim_R() const  { // current rank
            return XF[0]->getSize(1);
        }
        inline size_t dim_R_max() const  { // max allocated rank
            return XF[0]->getSize_max(1);
        }
        inline cudaError_t set_dim_R(const int dim_R_new, const cudaStream_t stream) { // set rank
            cudaError_t ce = (dim_R_new >=0 && dim_R_new <= dim_R_max()) ? cudaSuccess : cudaErrorInvalidValue;
            if(dim_R_new != dim_R()) {
                ce = lambda_v->resize(stream, -1, dim_R_new, -1);
                if(ce == cudaSuccess) {
                    ce = phi_d->resize(stream, -1, dim_R_new, -1);
                }
                for(int dd = 0; dd < XF.size() && ce == cudaSuccess; dd++) {
                    ce = XF[dd]->resize(stream, -1, dim_R_new, -1);
                }
                for(int dd = 0; dd < lambda_d.size() && ce == cudaSuccess; dd++) {
                    ce = lambda_d[dd]->resize(stream, -1, dim_R_new, -1);
                }
            }
            return ce;
        }
};


}; //namespace
#endif