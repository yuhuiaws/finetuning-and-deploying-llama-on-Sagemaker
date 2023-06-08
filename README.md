# finetuning-and-deploying-llama-on-Sagemaker

Use the two different methods (deepspeed and SageMaker model parallelism/SMP library) to fine tune llama model on Sagemaker. Then deploy the fine tuned llama on Sagemaker with server side batch. 

## There is four parts in this repo:
1. Prepare dataset: You can use the prepare-data-for-llama.ipynb and open source dataset such as dialy-dialogue.txt.zip to prepare dataset for llama.
2. Deploy fine tuned llama on SageMkaer: We use Large Model Inference/LMI container to deploy llama on SageMaker. Also, the demo code can perform the server side batch in order to improve the throughput. (The code is suitable for the case which is single sample/prompt per client request)
3. Fine tune llama by deepspeed on SageMaker multiple nodes: We use deepspeed and torch.distributed.launch to fine tune llama. 
4. Fine tune llama by SMP on SageMaker multiple nodes: We use SMP+HF trainer API to fine tune the llama, which is code zero intrusion.

## Tips:
* S5cmd should be used to download model and upload model in the training procedure, which will save much time.
* According my experiments, We should choose bf16 not fp16 for training LLM on A100 GPU, because bf16 has better training stability and convergence than fp16 (refer to: https://www.reddit.com/r/MachineLearning/comments/vndtn8/d_mixed_precision_training_difference_between/).
* The training speed between bf16 mixed precision training and fp16 mixed precision training is similar.
* When enabling RDMA protocol on EFA for P4d/P4de instance, there is very large improvement on deepspeed training speed. Just configure the following env variables in SageMaker SDK API: 'FI_PROVIDER': 'efa', 'NCCL_PROTO': 'simple', 'FI_EFA_USE_DEVICE_RDMA': '1' . For SMP 3D parallelism, the training speed has no any change when configuring the above env variables in SageMaker SDK API (I think SageMaker has enable the RDMA when using SMP 3D parallelism); also, even if we configure the above in the custom mpi option, the training speed has no any change.
* The training loss between deepspeed zero stage 1 and zero stage 3 is similar.
* The warmup step is very helpful for the convergence of the training loss, it is useful both for deepspeed training and SMP training.
* For my experiments and my datasets, when special new token such as [STOP] and [SEP] are added into the dataset, if both input embedding matrix and output embedding matrix are resized and are initialized by the mean pooling of others tokens’ embedding in corresponding embedding matrix (just like what the alpaca performs), the training procedure is unstable. Also, the convergence speed of train loss is slower than that of random initialization of the new tokens’ input embedding and output embedding.
* Deepspeed inference integrated by Large Model Inference/LMI container can support bf16 model, but the open source deepspeed inference does not support bf16 model (refer to:  https://github.com/microsoft/DeepSpeed/issues/2954 ).
* For text generation,  the length of input tokens is larger, the generation time is longer;the length of new generation tokens is larger, the generation time is longer; the main part of generation time results from the length of new generation tokens.
* When using HF pipeline API, batch inference/generation for pipeline API may increase or decrease the performance, which is up to the specific model, hardware, input tokens and output new tokens  (refer to https://huggingface.co/docs/transformers/main_classes/pipelines ). Also, from our experiments, for llama 7B fp16 model on g5.48xlarge:

      When input tokens is short such as 10,  the performance is better when setting the batch_size of pipeline API to be more than 1 (because the latency just becomes large a little and throughput is improved more).
      When input tokens is long such as 750,  the performance will become worse when setting the batch_size of pipeline API to be more than 1 (because the latency becomes very large compared with that of batch size 1).

So please test the performance case by case when configuring the batch_size parameter of HF pipeline API.

* For 7B/6B LLM fp16 model, g5.2xlarge has better performance-price ratio than g4dn.2xlarge.
* For 7B/6B LLM fp16/bf16 model, single GPU is better choice than multiple GPUs TP/PP. 
* If you want to deploy bf16 model on GPU instance, you should choose A10g or A100 instance (which is  Ampere architecture).
* You should trade off the performance and price when serving LLM model.  For the specific model size, 

    * Firstly you could evaluate whether single GPU can serve it. 
    * If not, you will choose multiple GPUs TP/PP.  Try fastertransformer first, then deepspped, finally HF accelerate. Just test the performance for the minimum number of GPUs as the start point.



