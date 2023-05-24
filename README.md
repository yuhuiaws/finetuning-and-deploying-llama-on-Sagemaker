# finetuning-and-deploying-llama-on-Sagemaker

Use the two different methods (deepspeed and SageMaker model parallelism/SMP library) to fine tune llama model on Sagemaker. Then deploy the fine tuned llama on Sagemaker with server side batch. 

## There is four parts in this repo:
1. Prepare dataset: You can use the prepare-data-for-llama.ipynb and open source dataset such as dialy-dialogue.txt.zip to prepare dataset for llama.
2. Deploy fine tuned llama on SageMkaer: We use Large Model Inference/LMI container to deploy llama on SageMaker. Also, the demo code can perform the server side batch in order to improve the throughput. (The code is suitable for the case which is single sample/prompt per client request)
3. Fine tune llama by deepspeed on SageMaker multiple nodes: We use deepspeed and torch.distributed.launch to fine tune llama. 
4. Fine tune llama by SMP on SageMaker multiple nodes: We use SMP+HF trainer API to fine tune the llama, which is code zero intrusion.

## Tips:
* S5cmd should be used to download model and upload model in the training procedure, which will save much time.
* We should choose bf16 not fp16 for training LLM on A100 GPU, because bf16 has better training stability and convergence than fp16.
* The training speed between bf16 mixed precision training and fp16 mixed precision training is similar.
* The training loss between deepspeed zero stage 1 and zero stage 3 is similar.
* The warmup step is very helpful for the convergence of the training loss, it is useful both for deepspeed training and SMP training.
* For my experiments and my datasets, when special new token such as [STOP] and [SEP] are added into the dataset, if both input embedding matrix and output embedding matrix are resized and are initialized by the mean pooling of others tokens’ embedding in corresponding embedding matrix (just like what the alpaca performs), the training procedure is unstable.
* Deepspeed inference integrated by Large Model Inference/LMI container can support bf16 model, but the open source deepspeed inference does not support bf16 model (refer to:  https://github.com/microsoft/DeepSpeed/issues/2954 ).
* For text generation,  the main part of generation time results from the length of new generation tokens.
* 


